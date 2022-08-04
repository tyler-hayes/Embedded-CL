import os
import torch
from torch import nn


class NaiveBayes(nn.Module):
    """
    This is an implementation of the Gaussian Naive Bayes algorithm for streaming learning.
    """

    def __init__(self, input_shape, num_classes, backbone=None, device='cuda', shrinkage_param=1e-4):
        """
        Init function for the model.
        :param input_shape: feature dimension
        :param num_classes: number of total classes in stream
        """

        super(NaiveBayes, self).__init__()

        # parameters
        self.device = device
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.shrinkage_param = shrinkage_param

        # feature extraction backbone
        self.backbone = backbone
        if backbone is not None:
            self.backbone = backbone.eval().to(device)

        # setup weights
        self.num_updates = 0
        self.prev_num_updates = 0
        self.muK = torch.zeros((num_classes, input_shape)).to(self.device)
        self.muK2 = torch.zeros((num_classes, input_shape)).to(self.device)
        self.varK = torch.zeros((num_classes, input_shape)).to(self.device)
        self.cK = torch.zeros(num_classes).to(self.device)
        self.Lambda = []

    @torch.no_grad()
    def fit(self, x, y, item_ix):
        """
        Fit the model to a new sample (x,y).
        :param item_ix:
        :param x: a torch tensor of the input data (must be a vector)
        :param y: a torch tensor of the input label
        :return: None
        """
        x = x.to(self.device)
        y = y.long().to(self.device)

        # make sure things are the right shape
        if len(x.shape) < 2:
            x = x.unsqueeze(0)
        if len(y.shape) == 0:
            y = y.unsqueeze(0)

        self.num_updates += 1
        self.cK[y] += 1
        n = self.cK[y]
        delta = (x - self.muK[y, :])
        self.muK[y, :] += delta / n

        # variance update (https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Incremental_algorithm)
        self.muK2[y, :] += delta * (x - self.muK[y, :])

        if n >= 2:
            self.varK[y, :] = self.muK2[y, :] / (n - 1)

    @torch.no_grad()
    def predict(self, X, return_probas=False):
        """
        Make predictions on test data X.
        :param X: a torch tensor that contains N data samples (N x d)
        :param return_probas: True if the user would like probabilities instead of predictions returned
        :return: the test predictions or probabilities
        """
        X = X.to(self.device)

        with torch.no_grad():
            # initialize parameters for testing

            # compute/load Lambda matrix
            if self.prev_num_updates != self.num_updates:
                # there have been updates to the model, compute Lambda
                # print('\nFirst predict since model update...computing Lambda matrix...')
                l = []
                for k in range(self.num_classes):
                    varK = torch.eye(self.input_shape).to(self.device) * self.varK[k]
                    Lambda = torch.pinverse(
                        (1 - self.shrinkage_param) * varK + self.shrinkage_param * torch.eye(self.input_shape).to(
                            self.device))
                    l.append(Lambda)
                self.Lambda = l
                self.prev_num_updates = self.num_updates
            else:
                l = self.Lambda

            scores = torch.empty((X.shape[0], self.num_classes))

            for k in range(self.num_classes):
                W = torch.matmul(l[k], self.muK[k, :])
                c = 0.5 * torch.sum(self.muK[k, :] * W, dim=0)
                scores[:, k] = torch.matmul(X, W) - c

            not_visited_ix = torch.where(self.cK == 0)[0]
            min_col = torch.min(scores, dim=1)[0].unsqueeze(0) - 1
            scores[:, not_visited_ix] = min_col.tile(len(not_visited_ix)).reshape(
                len(not_visited_ix), len(X)).transpose(1, 0)  # mask off scores for unseen classes

            # return predictions or probabilities
            if not return_probas:
                return scores.cpu()
            else:
                return torch.softmax(scores, dim=1).cpu()

    @torch.no_grad()
    def ood_predict(self, x):
        return self.predict(x, return_probas=True)

    @torch.no_grad()
    def evaluate_ood_(self, test_loader):
        print('\nTesting OOD on %d images.' % len(test_loader.dataset))

        num_samples = len(test_loader.dataset)
        scores = torch.empty((num_samples, self.num_classes))
        labels = torch.empty(num_samples).long()
        start = 0
        for test_x, test_y in test_loader:
            if self.backbone is not None:
                batch_x_feat = self.backbone(test_x.to(self.device))
            else:
                batch_x_feat = test_x.to(self.device)
            ood_scores = self.ood_predict(batch_x_feat)
            end = start + ood_scores.shape[0]
            scores[start:end] = ood_scores
            labels[start:end] = test_y.squeeze()
            start = end

        return scores, labels

    @torch.no_grad()
    def fit_batch(self, batch_x, batch_y, batch_ix):
        # fit one example at a time
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
    def evaluate_(self, test_loader):
        print('\nTesting on %d images.' % len(test_loader.dataset))

        num_samples = len(test_loader.dataset)
        probabilities = torch.empty((num_samples, self.num_classes))
        labels = torch.empty(num_samples).long()
        start = 0
        for test_x, test_y in test_loader:
            if self.backbone is not None:
                batch_x_feat = self.backbone(test_x.to(self.device))
            else:
                batch_x_feat = test_x.to(self.device)
            probas = self.predict(batch_x_feat, return_probas=True)
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
        d['muK'] = self.muK.cpu()
        d['muK2'] = self.muK2.cpu()
        d['varK'] = self.varK.cpu()
        d['cK'] = self.cK.cpu()
        d['num_updates'] = self.num_updates

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
        self.muK = d['muK'].to(self.device)
        self.muK2 = d['muK2'].to(self.device)
        self.varK = d['varK'].to(self.device)
        self.cK = d['cK'].to(self.device)
        self.num_updates = d['num_updates']
