import os
import torch
from torch import nn
import numpy as np
from collections import defaultdict


class CBCL(nn.Module):
    """
    This is an implementation of the CBCL algorithm.
    """

    def __init__(self, input_shape, num_classes, backbone=None, device='cuda', buffer_size=200,
                 distance_threshold=150, cluster_removal_approach='min_dist', topk=5, weighted_pred=True):
        """
        Init function for the CBCL model.
        :param input_shape: feature dimension
        :param num_classes: number of total classes in stream
        """

        super(CBCL, self).__init__()

        # parameters
        self.device = device
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.buffer_size = buffer_size
        self.distance_threshold = distance_threshold
        self.cluster_removal_approach = cluster_removal_approach
        self.topk = topk
        self.weighted_pred = weighted_pred

        # feature extraction backbone
        self.backbone = backbone
        if backbone is not None:
            self.backbone = backbone.eval().to(device)

        # setup weights
        self.cluster_dict = defaultdict(list)  # key is label, value is list of tuple (counts, clusters)
        self.min_dist_dict = dict()  # to maintain minimum distances between class clusters to save time
        self.cluster_count = 0
        self.num_updates = 0

    @torch.no_grad()
    def fit(self, x, y, item_ix):
        """
        Fit the CBCL model to a new sample (x,y).
        :param item_ix:
        :param x: a torch tensor of the input data (must be a vector)
        :param y: a torch tensor of the input label
        :return: None
        """
        x = x.to(self.device)
        y = y.long().item()  # int of label

        # make sure things are the right shape
        if len(x.shape) < 2:
            x = x.unsqueeze(0)

        # cluster updates
        if len(self.cluster_dict[y]) == 0:
            # class not yet visited, store sample
            self.cluster_dict[y].append([1.])  # count is 1
            self.cluster_dict[y].append([x])  # store raw sample
            self.min_dist_dict[y] = (np.inf, 0, 0)  # min distance, ix1, ix2
            self.cluster_count += 1.
        else:
            curr_class_counts, curr_class_clusters = self.cluster_dict[y]
            curr_class_clusters_tensor = torch.cat(curr_class_clusters, dim=0)  # put into tensor
            inds, dists = self.find_min_dists(curr_class_clusters_tensor, x)
            # find the closest cluster
            closest_ix = torch.argmin(dists)
            closest_dist = torch.min(dists)
            if closest_dist < self.distance_threshold:
                # close to existing cluster, merge
                exist_cluster_count = curr_class_counts[closest_ix]
                exist_cluster = curr_class_clusters[closest_ix]
                new_cluster = (exist_cluster_count * exist_cluster + x) / (exist_cluster_count + 1)
                new_count = exist_cluster_count + 1.

                # put new cluster into list
                curr_class_counts[closest_ix] = new_count
                curr_class_clusters[closest_ix] = new_cluster
            else:
                # far from existing clusters, add new point
                self.cluster_dict[y][0].append(1.)
                self.cluster_dict[y][1].append(x)
                self.cluster_count += 1

            self.update_min_dist_dict(y, curr_class_clusters)

        # merge clusters if there are more than the buffer allows
        if self.cluster_count > self.buffer_size:
            # remove a cluster
            if self.cluster_removal_approach == 'max':
                # merge two clusters from class with most clusters
                curr_max = -1
                ix1 = np.inf
                ix2 = np.inf
                class_ = np.inf
                for class_label, (dist_, ix1_, ix2_) in self.min_dist_dict.items():
                    class_sum = sum(self.cluster_dict[class_label][0])  # find number of clusters for associated class
                    if class_sum > curr_max:
                        curr_max = class_sum
                        ix1 = ix1_
                        ix2 = ix2_
                        class_ = class_label

                self.merge_two_closest_clusters(class_, ix1, ix2)
            elif self.cluster_removal_approach == 'min_dist':
                # merge two clusters from class with two closest clusters
                curr_min = np.inf
                ix1 = np.inf
                ix2 = np.inf
                class_ = np.inf
                for class_label, (dist_, ix1_, ix2_) in self.min_dist_dict.items():
                    if dist_ < curr_min:
                        curr_min = dist_
                        ix1 = ix1_
                        ix2 = ix2_
                        class_ = class_label
                self.merge_two_closest_clusters(class_, ix1, ix2)
            else:
                raise NotImplementedError

            self.cluster_count -= 1

        self.num_updates += 1

    @torch.no_grad()
    def merge_two_closest_clusters(self, class_, ix1, ix2):
        # merge two closest clusters using Ward's algorithm
        curr_class_counts, curr_class_clusters = self.cluster_dict[class_]
        cluster1 = curr_class_clusters[ix1]
        cluster2 = curr_class_clusters[ix2]
        count1 = curr_class_counts[ix1]
        count2 = curr_class_counts[ix2]
        new_count = count1 + count2
        new_cluster = (count1 * cluster1 + count2 * cluster2) / new_count
        curr_class_clusters[ix2] = new_cluster
        curr_class_counts[ix2] = new_count
        curr_class_clusters.pop(ix1)
        curr_class_counts.pop(ix1)
        self.update_min_dist_dict(class_, curr_class_clusters)

    @torch.no_grad()
    def update_min_dist_dict(self, y, curr_class_clusters):
        # check that we have more than one cluster for this class to find two closest clusters
        if len(self.cluster_dict[y][0]) > 1:
            # find the two closest clusters for the class since clusters were updated
            curr_class_clusters_tensor = torch.cat(curr_class_clusters, dim=0)  # put into tensor
            ix1, ix2, min_dist = self.find_closest_clusters(curr_class_clusters_tensor)
            self.min_dist_dict[y] = (min_dist, ix1, ix2)
        else:
            self.min_dist_dict[y] = (np.inf, 0, 0)

    @torch.no_grad()
    def find_closest_clusters(self, H):
        """
        Given an array of data, compute the indices of the two closest samples.
        :param H: an Nxd array of data (PyTorch Tensor)
        :return: the two indices of the closest samples
        """
        psi = self.find_dists(H, H)

        # infinity on diagonal
        psi.fill_diagonal_(np.inf)

        # grab indices
        dist = torch.min(psi)
        idx_row, idx_col = np.unravel_index(torch.argmin(psi).cpu(), psi.shape)
        return min(idx_row, idx_col), max(idx_row, idx_col), dist

    @torch.no_grad()
    def find_min_dists(self, A, B):
        """
        Given a matrix of points A, return the indices of the closest points in A to B using L2 distance.
        :param A: N x d matrix of points
        :param B: M x d matrix of points for predictions
        :return: (indices, distances) of closest points in A
        """
        dist = self.find_dists(A, B)
        dists, inds = torch.min(dist, dim=1)
        return inds, dists

    @torch.no_grad()
    def find_dists(self, A, B):
        """
        Given a matrix of points A, return the indices of the closest points in A to B using L2 distance.
        :param A: N x d matrix of points
        :param B: M x d matrix of points for predictions
        :return: distance matrix of size (MxN)
        """
        dist = torch.cdist(B, A)
        return dist

    @torch.no_grad()
    def predict(self, X, return_probas=False, clusters=None, counts=None, labels=None):
        """
        Make predictions on test data X.
        :param X: a torch tensor that contains N data samples (N x d)
        :param return_probas: True if the user would like probabilities instead of predictions returned
        :return: the test predictions or probabilities
        """
        X = X.to(self.device)
        mb_size = len(X)

        # compute distance of every x to all clusters
        k = min(self.topk, len(clusters))
        dist_mat = self.find_dists(clusters, X)  # size of dist_mat is N x num_clusters
        topk_dists, topk_ixs = torch.topk(-dist_mat, k, dim=1)  # find top-k closest clusters for each x
        topk_dists = -topk_dists
        topk_dists = 1 / topk_dists  # invert distances

        # initialize scores to zeros
        topk_ixs_flat = topk_ixs.view((mb_size * k))
        labels_ixs = labels[topk_ixs_flat].view((mb_size, k))
        expanded_dists = topk_dists.unsqueeze(-1).expand(mb_size, k, self.num_classes)
        one_hot = torch.nn.functional.one_hot(labels_ixs.long(), num_classes=self.num_classes)

        scores = torch.sum(expanded_dists.to(self.device) * one_hot.to(self.device), dim=1)
        if self.weighted_pred:
            scores *= counts.to(self.device)

        # mask off predictions for unseen classes
        not_visited_ix = torch.where(counts == 0)[0]
        min_col = torch.min(scores, dim=1)[0].unsqueeze(0) - 1
        scores[:, not_visited_ix] = min_col.tile(len(not_visited_ix)).reshape(
            len(not_visited_ix), len(X)).transpose(1, 0)  # mask off scores for unseen classes

        # return predictions or probabilities
        if not return_probas:
            return scores.cpu()
        else:
            return torch.softmax(scores, dim=1).cpu()

    @torch.no_grad()
    def fit_batch(self, batch_x, batch_y, batch_ix):
        # fit CBCL one example at a time
        for x, y in zip(batch_x, batch_y):
            self.fit(x.cpu(), y.view(1, ), None)

    @torch.no_grad()
    def train_(self, train_loader):
        # print('\nTraining on %d images.' % len(train_loader.dataset))

        for batch_x, batch_y, batch_ix in train_loader:
            if self.backbone is not None:
                batch_x_feat = self.backbone(batch_x.to(self.device))
            else:
                batch_x_feat = batch_x.to(self.device)

            self.fit_batch(batch_x_feat, batch_y, batch_ix)

    @torch.no_grad()
    def gather_clusters(self):
        # put class clusters into lists of concatenated tensors
        cluster_list = []
        labels = []
        for k, v in self.cluster_dict.items():
            cluster_list.append(torch.cat(v[1], dim=0))
            counts_ = torch.tensor(v[0])
            labels.append(k * torch.ones_like(counts_))

        count_list = []
        for i in range(self.num_classes):
            if i in self.cluster_dict:
                v = self.cluster_dict[i]
                counts_ = torch.tensor(v[0])
                sum_count = sum(counts_)
                count_list.append(sum_count)
            else:
                count_list.append(0)

        clusters = torch.cat(cluster_list, dim=0)  # length is num_clusters
        counts = torch.tensor(count_list)  # length is num_classes
        labels = torch.cat(labels)  # length is num_clusters
        return clusters, counts, labels

    @torch.no_grad()
    def evaluate_(self, test_loader):
        print('\nTesting on %d images.' % len(test_loader.dataset))

        clusters_, counts_, labels_ = self.gather_clusters()

        num_samples = len(test_loader.dataset)
        probabilities = torch.empty((num_samples, self.num_classes))
        labels = torch.empty(num_samples).long()
        start = 0
        for test_x, test_y in test_loader:
            if self.backbone is not None:
                batch_x_feat = self.backbone(test_x.to(self.device))
            else:
                batch_x_feat = test_x.to(self.device)
            probas = self.predict(batch_x_feat, return_probas=True, clusters=clusters_, counts=counts_,
                                  labels=labels_)
            end = start + probas.shape[0]
            probabilities[start:end] = probas
            labels[start:end] = test_y.squeeze()
            start = end
        return probabilities, labels

    def save_model(self, save_path, save_name):
        """
        Save the model parameters to a torch file.
        :param save_path: the path where the model will be saved
        :param save_name: the name for the saved file
        :return:
        """
        # grab parameters for saving
        d = dict()
        # TODO: fill in

        # save model out
        torch.save(d, os.path.join(save_path, save_name + '.pth'))

    def load_model(self, save_file):
        """
        Load the model parameters into StreamingLDA object.
        :param save_path: the path where the model is saved
        :param save_name: the name of the saved file
        :return:
        """
        # load parameters
        print('\nloading ckpt from: %s' % save_file)
        d = torch.load(os.path.join(save_file))
        # TODO: fill in
