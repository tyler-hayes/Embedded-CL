import os
import time

import torch
from torch import nn
import torch.nn.functional as F

from utils import get_logger, TFLogger, AverageMeter, ProgressMeter, accuracy, save_checkpoint, makedirs


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


class OfflineSoftmax(nn.Module):
    """
    This is an implementation of the Softmax algorithm.
    """

    def __init__(self, args, input_shape, num_classes, backbone=None, device='cuda', lr=0.01,
                 weight_decay=1e-4, lr_milestones=[15, 30], epochs=10, print_freq=100, save_dir=None):

        super(OfflineSoftmax, self).__init__()

        makedirs(save_dir)

        # parameters
        self.device = device
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.print_freq = print_freq

        # feature extraction backbone
        self.backbone = backbone
        if backbone is not None:
            self.backbone = backbone.eval().to(device)

        self.linear = SoftmaxLayer(input_shape, num_classes)
        self.linear = self.linear.to(device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.linear.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        self.epochs = epochs
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=lr_milestones)

        # logging
        self.logger = get_logger(logpath=os.path.join(save_dir, 'logs'), filepath=os.path.abspath(__file__))
        self.logger.info(args)

        tf_dir = os.path.join(save_dir, 'tf_logging')
        self.tf_logger = TFLogger(tf_dir)

    def train_and_eval(self, train_loader, val_loader):
        print('\nstarting training...')
        best_acc1 = 0
        for epoch in range(self.epochs):
            # train for one epoch
            self.one_epoch(train_loader, epoch)

            # evaluate on validation set
            acc1 = self.validate(val_loader, epoch)

            # modify lr
            self.lr_scheduler.step()
            self.logger.info('LR: {:f}'.format(self.lr_scheduler.get_last_lr()[-1]))

            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.linear.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict(),
            }, is_best, self.save_dir)

    def one_epoch(self, train_loader, epoch):
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses, top1, top5],
            prefix="Epoch: [{}]".format(epoch))

        # switch to train mode
        self.linear.train()

        end = time.time()
        for i, (feature, target) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            # compute output
            if self.backbone is not None:
                feature_ = self.backbone(feature.to(self.device))
            else:
                feature_ = feature.to(self.device)

            output = self.linear(feature_)
            target = target.to(self.device)
            loss = F.cross_entropy(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), feature_.size(0))
            top1.update(acc1[0], feature_.size(0))
            top5.update(acc5[0], feature_.size(0))
            self.tf_logger.add_scalar('train_loss', losses.avg, epoch)
            self.tf_logger.add_scalar('train_top1', top1.avg, epoch)
            self.tf_logger.add_scalar('train_top5', top5.avg, epoch)

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.print_freq == 0:
                self.logger.info(progress.display(i))

    def validate(self, val_loader, epoch):
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(
            len(val_loader),
            [batch_time, losses, top1, top5],
            prefix='Test: ')

        self.linear.eval()

        with torch.no_grad():
            end = time.time()
            for i, (feature, target) in enumerate(val_loader):

                # compute output
                if self.backbone is not None:
                    feature_ = self.backbone(feature.to(self.device))
                else:
                    feature_ = feature.to(self.device)

                output = self.linear(feature_)
                target = target.to(self.device)
                loss = F.cross_entropy(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), feature_.size(0))
                top1.update(acc1[0], feature_.size(0))
                top5.update(acc5[0], feature_.size(0))
                self.tf_logger.add_scalar('val_loss', losses.avg, epoch)
                self.tf_logger.add_scalar('val_top1', top1.avg, epoch)
                self.tf_logger.add_scalar('val_top5', top5.avg, epoch)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % self.print_freq == 0:
                    self.logger.info(progress.display(i))

            # TODO: this should also be done with the ProgressMeter
            self.logger.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                             .format(top1=top1, top5=top5))

        return top1.avg

    @torch.no_grad()
    def predict(self, data_loader):

        probas = torch.zeros((len(data_loader.dataset), self.num_classes))
        all_lbls = torch.zeros((len(data_loader.dataset)))
        start_ix = 0
        for batch_ix, batch in enumerate(data_loader):
            feature, batch_lbls = batch[0], batch[1]
            batch_lbls = batch_lbls

            if self.backbone is not None:
                feature_ = self.backbone(feature.to(self.device))
            else:
                feature_ = feature.to(self.device)

            output = self.linear(feature_)

            end_ix = start_ix + len(feature_)
            probas[start_ix:end_ix] = torch.softmax(output, dim=1).cpu()
            all_lbls[start_ix:end_ix] = batch_lbls.squeeze()
            start_ix = end_ix

        preds = probas.max(1)[1]

        return preds, probas, all_lbls.long()
