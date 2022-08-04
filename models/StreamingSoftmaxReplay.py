from collections import defaultdict
import torch
from torch import nn
import random
import numpy as np
import os

from utils import randint, CMA


class SoftmaxLayer(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(SoftmaxLayer, self).__init__()

        linear = torch.nn.Linear(input_size, output_size)
        linear.weight.data.normal_(mean=0.0, std=0.01)
        linear.bias.data.zero_()
        self.fc = linear

    def forward(self, x):
        out = self.fc(x)
        return out


class StreamingSoftmax(nn.Module):
    """
    This is an implementation of the Streaming Softmax algorithm for streaming learning.
    """

    def __init__(self, input_shape, num_classes, use_replay=False, backbone=None, device='cuda', lr=0.1,
                 weight_decay=1e-5, replay_samples=50, max_buffer_size=7300):
        """
        Init function for the Streaming Softmax model.
        :param input_shape: feature dimension
        :param num_classes: number of total classes in stream
        """

        super(StreamingSoftmax, self).__init__()

        # parameters
        self.device = device
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.use_replay = use_replay
        self.replay_samples = replay_samples
        self.max_buffer_size = max_buffer_size

        # feature extraction backbone
        self.backbone = backbone
        if backbone is not None:
            self.backbone = backbone.eval().to(device)

        # model specific structures
        self.latent_dict = {}
        self.rehearsal_ixs = []
        self.class_id_to_item_ix_dict = defaultdict(list)
        self.num_updates = 0
        self.total_loss = CMA()
        self.msg = '\rSample %d -- train_loss=%1.6f -- buffer_size=%d'
        self.cK = torch.zeros(num_classes).to(device)

        self.classifier = SoftmaxLayer(input_shape, num_classes)
        self.classifier = self.classifier.to(device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.classifier.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

    def fit_fine_tune(self, x, y):
        self.classifier.train()

        # zero out grads before backward pass because they are accumulated
        self.optimizer.zero_grad()

        data_points = torch.unsqueeze(x, 0).to(self.device)
        data_labels = y.to(self.device)

        output = self.classifier(data_points)
        loss = self.criterion(output, data_labels)
        loss.backward()
        self.optimizer.step()

        self.total_loss.update(loss.item())

        print(self.msg % (self.num_updates, self.total_loss.avg, len(self.rehearsal_ixs)), end="")
        self.cK[y] += 1
        self.num_updates += 1

    def fit_replay(self, x, y, item_ix):
        self.classifier.train()

        # zero out grads before backward pass because they are accumulated
        self.optimizer.zero_grad()
        num_samples_in_buffer = len(self.rehearsal_ixs)

        if num_samples_in_buffer == 0:
            data_points = torch.unsqueeze(x, 0).to(self.device)
            data_labels = y.to(self.device)
        elif num_samples_in_buffer < self.replay_samples:  # if buffer less than self.replay_samples
            num_samples = num_samples_in_buffer
            data_points = torch.empty((num_samples + 1, self.input_shape)).to(self.device)
            data_labels = torch.empty((num_samples + 1), dtype=torch.long).to(self.device)
            data_points[0] = x.to(self.device)
            data_labels[0] = y.to(self.device)
            ixs = list(np.arange(len(self.rehearsal_ixs)))
            ixs = [self.rehearsal_ixs[_curr_ix] for _curr_ix in ixs]
            for ii, v in enumerate(ixs):
                data_points[ii + 1] = torch.from_numpy(self.latent_dict[v][0]).to(self.device)
                data_labels[ii + 1] = torch.from_numpy(self.latent_dict[v][1]).to(self.device)
        else:  # if buffer more than self.replay_samples
            num_samples = self.replay_samples
            data_points = torch.empty((num_samples + 1, self.input_shape)).to(self.device)
            data_labels = torch.empty((num_samples + 1), dtype=torch.long).to(self.device)
            data_points[0] = x.to(self.device)
            data_labels[0] = y.to(self.device)
            ixs = randint(len(self.rehearsal_ixs), num_samples)
            ixs = [self.rehearsal_ixs[_curr_ix] for _curr_ix in ixs]
            for ii, v in enumerate(ixs):
                data_points[ii + 1] = torch.from_numpy(self.latent_dict[v][0]).to(self.device)
                data_labels[ii + 1] = torch.from_numpy(self.latent_dict[v][1]).to(self.device)

        output = self.classifier(data_points)
        loss = self.criterion(output, data_labels)
        loss.backward()
        self.optimizer.step()

        self.total_loss.update(loss.item())

        print(self.msg % (self.num_updates, self.total_loss.avg, len(self.rehearsal_ixs)), end="")
        self.num_updates += 1
        self.cK[y] += 1

        item_ix_np = int(item_ix.cpu().numpy())
        label_np = y.cpu().numpy()
        data_np = x.cpu().numpy()

        # add new instance to buffer
        self.latent_dict[item_ix_np] = [data_np, label_np]
        self.rehearsal_ixs.append(item_ix_np)
        self.class_id_to_item_ix_dict[int(label_np)].append(item_ix_np)

        # if buffer is full, randomly replace previous example from class with most samples
        if len(self.rehearsal_ixs) >= self.max_buffer_size:
            # class with most samples and random item_ix from it
            max_key = max(self.class_id_to_item_ix_dict, key=lambda x: len(self.class_id_to_item_ix_dict[x]))
            max_class_list = self.class_id_to_item_ix_dict[max_key]
            rand_item_ix = random.choice(max_class_list)

            # remove the random_item_ix from all buffer references
            max_class_list.remove(rand_item_ix)
            self.latent_dict.pop(rand_item_ix)
            self.rehearsal_ixs.remove(rand_item_ix)

    @torch.no_grad()
    def ood_predict(self, x):
        return self.predict(x, return_probas=True)

    @torch.no_grad()
    def predict(self, X, return_probas=False):
        self.classifier.eval()
        X = X.to(self.device)
        scores = self.classifier(X)

        # mask off predictions for unseen classes
        not_visited_ix = torch.where(self.cK == 0)[0]
        min_col = torch.min(scores, dim=1)[0].unsqueeze(0) - 1
        scores[:, not_visited_ix] = min_col.tile(len(not_visited_ix)).reshape(
            len(not_visited_ix), len(X)).transpose(1, 0)  # mask off scores for unseen classes

        # return predictions or probabilities
        if not return_probas:
            return scores.cpu()
        else:
            return torch.softmax(scores, dim=1).cpu()

    def fit(self, x, y, item_ix):
        if self.use_replay:
            self.fit_replay(x, y, item_ix)
        else:
            self.fit_fine_tune(x, y)

    def fit_batch(self, batch_x, batch_y, batch_ix):
        # fit model one example at a time
        for x, y, item_ix in zip(batch_x, batch_y, batch_ix):
            self.fit(x, y.view(1, ), item_ix)

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

    def save_model(self, save_path, save_name):
        """
        Save the model parameters to a torch file.
        :param save_path: the path where the model will be saved
        :param save_name: the name for the saved file
        :return:
        """

        state = {
            'state_dict': self.classifier.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(state, os.path.join(save_path, save_name + '.pth'))

    def load_model(self, save_file):
        """
        Load the model parameters into StreamingLDA object.
        :param save_path: the path where the model is saved
        :param save_name: the name of the saved file
        :return:
        """
        d = torch.load(os.path.join(save_file))
        print('\nloading ckpt from: %s' % save_file)
        self.classifier.load_state_dict(d['state_dict'])
        self.optimizer.load_state_dict(d['optimizer'])
