from argparse import ArgumentParser
import numpy as np
from time import time
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms as tf
from tqdm import tqdm

from torchplus.nn import extra as ex


def parse_flags():
    a = ArgumentParser()

    # Hardware.
    a.add_argument('--use-cuda', type=int, default=1)

    # Training.
    a.add_argument('--seed', type=int, default=0x31337)
    a.add_argument('--num-epochs', type=int, default=5)
    a.add_argument('--batch-size', type=int, default=32)

    return a.parse_args()


class Loader(object):
    def __init__(self, use_cuda, batch_size):
        self.use_cuda = use_cuda
        self.batch_size = batch_size

        dirname = '../data/'
        t = tf.Compose([
            tf.ToTensor(),
            tf.Normalize((0.1307,), (0.3081,)),
        ])
        train_dataset = MNIST(dirname, train=True, download=True, transform=t)
        t = tf.Compose([
            tf.ToTensor(),
            tf.Normalize((0.1307,), (0.3081,)),
        ])
        val_dataset = MNIST(dirname, train=False, download=False, transform=t)

        kwargs = {
            'batch_size': batch_size,
            'shuffle': True,
        }
        if use_cuda:
            kwargs.update({
                'num_workers': 1,
                'pin_memory': True,
            })
        self.train = DataLoader(train_dataset, **kwargs)
        self.val = DataLoader(val_dataset, **kwargs)

        self.num_batches = len(self.train) + len(self.val)

    def each_batch(self):
        ones = np.ones(len(self.train), 'uint8')
        zeros = np.zeros(len(self.val), 'uint8')
        splits = np.concatenate([ones, zeros])
        np.random.shuffle(splits)
        each_train = iter(self.train)
        each_val = iter(self.val)
        for is_training in splits:
            if is_training:
                each_split = each_train
            else:
                each_split = each_val
            images, labels = next(each_split)
            if self.use_cuda:
                images = images.cuda()
                labels = labels.cuda()
            yield is_training, images, labels

    def each_batch_tqdm(self):
        for x in tqdm(self.each_batch(), total=self.num_batches, leave=False):
            yield x


class ImageClassifier(nn.Module):
    def train_on_batch(self, images, labels, optimizer):
        self.train()
        optimizer.zero_grad()
        preds = self.forward(images)
        loss = F.nll_loss(preds, labels)
        loss.backward()
        loss = np.asscalar(loss.detach().cpu().numpy())
        optimizer.step()
        correct = preds.max(1)[1] == labels
        acc = correct.sum().type(torch.float32) / len(correct)
        acc = np.asscalar(acc.cpu().numpy())
        return loss, acc

    def val_on_batch(self, images, labels):
        self.eval()
        preds = self.forward(images)
        loss = F.nll_loss(preds, labels)
        loss = np.asscalar(loss.detach().cpu().numpy())
        correct = preds.max(1)[1] == labels
        acc = correct.sum().type(torch.float32) / len(correct)
        acc = np.asscalar(acc.cpu().numpy())
        return loss, acc

    def fit_on_epoch(self, loader, optimizer):
        train_losses = []
        train_accs = []
        train_times = []
        val_losses = []
        val_accs = []
        val_times = []
        for is_training, images, labels in loader.each_batch_tqdm():
            if is_training:
                t0 = time()
                loss, acc = self.train_on_batch(images, labels, optimizer)
                t = time() - t0
                train_losses.append(loss)
                train_accs.append(acc)
                train_times.append(t)
            else:
                t0 = time()
                loss, acc = self.val_on_batch(images, labels)
                t = time() - t0
                val_losses.append(loss)
                val_accs.append(acc)
                val_times.append(t)
        train = np.mean(train_losses), np.mean(train_accs), np.mean(train_times)
        val = np.mean(val_losses), np.mean(val_accs), np.mean(val_times)
        return train, val

    def fit(self, num_epochs, loader, optimizer):
        train_losses = []
        train_accs = []
        train_times = []
        val_losses = []
        val_accs = []
        val_times = []
        for epoch in range(num_epochs):
            train, val = self.fit_on_epoch(loader, optimizer)
            print('%d acc %.2f/%.2f%% loss %.4f/%.4f time %.3f/%.3fms' %
                  (epoch, train[1] * 100, val[1] * 100, train[0], val[0],
                   train[2] * 1000, val[2] * 1000))
            if epoch:
                train_loss, train_acc, train_time = train
                train_losses.append(train_loss)
                train_accs.append(train_acc)
                train_times.append(train_time)
                val_loss, val_acc, val_time = val
                val_losses.append(val_loss)
                val_accs.append(val_acc)
                val_times.append(val_time)
        index = np.array(val_losses).argmin()
        train_loss = train_losses[index]
        train_acc = train_accs[index]
        train_time = np.mean(train_times)
        train = train_loss, train_acc, train_time
        val_loss = val_losses[index]
        val_acc = val_accs[index]
        val_time = np.mean(val_times)
        val = val_loss, val_acc, val_time
        print('best: %d %.2f/%.2f%%; time %.3f/%.3fms' %
              (index + 1, train_acc * 100, val_acc * 100, train_time * 1000,
               val_time * 1000))
        return train, val


class OrigModel(ImageClassifier):
    desc = 'Model written in PyTorch Classic (from PyTorch examples repo).'

    def __init__(self):
        from torch import nn
        ImageClassifier.__init__(self)
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class SequentialModel(ImageClassifier):
    desc = 'Rewritten to use nn.Sequential, for comparison.'

    def __init__(self):
        from torch import nn
        ImageClassifier.__init__(self)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 20, kernel_size=5),
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(320, 50),
            nn.ReLU(),
            nn.Dropout(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(50, 10),
            nn.LogSoftmax(1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 320)
        x = self.fc1(x)
        return self.fc2(x)


class ExtraModel(ImageClassifier):
    desc = 'With nn.Sequential and a TorchPlus extra (ex.Flatten).'

    def __init__(self):
        from torch import nn
        ImageClassifier.__init__(self)
        conv1 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        conv2 = nn.Sequential(
            nn.Conv2d(10, 20, kernel_size=5),
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        fc1 = nn.Sequential(
            nn.Linear(320, 50),
            nn.ReLU(),
            nn.Dropout(),
        )
        fc2 = nn.Sequential(
            nn.Linear(50, 10),
            nn.LogSoftmax(1)
        )
        self.seq = nn.Sequential(conv1, conv2, ex.Flatten(), fc1, fc2)

    def forward(self, x):
        return self.seq(x)


class TorchPlusModel(ImageClassifier):
    desc = 'Written using TorchPlus "+" sequences.'

    def __init__(self):
        from torchplus import nn
        ImageClassifier.__init__(self)
        conv1 = nn.Conv2d(1, 10, kernel_size=5) + nn.MaxPool2d(2) + nn.ReLU
        conv2 = nn.Conv2d(10, 20, kernel_size=5) + nn.Dropout2d() + \
            nn.MaxPool2d(2) + nn.ReLU
        fc1 = nn.Linear(320, 50) + nn.ReLU + nn.Dropout
        fc2 = nn.Linear(50, 10) + nn.LogSoftmax(1)
        self.seq = conv1 + conv2 + nn.Flatten + fc1 + fc2

    def forward(self, x):
        return self.seq(x)


def run(flags):
    loader = Loader(flags.use_cuda, flags.batch_size)
    classes = OrigModel, SequentialModel, ExtraModel, TorchPlusModel
    for klass in classes:
        np.random.seed(flags.seed)
        torch.manual_seed(flags.seed)
        print(klass.desc)
        model = klass()
        if flags.use_cuda:
            model = model.cuda()
        optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
        model.fit(flags.num_epochs, loader, optimizer)
        print()


if __name__ == '__main__':
    run(parse_flags())
