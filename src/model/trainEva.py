import torch
import torch.nn as nn
import numpy as np


class TrainEva:
    def __init__(self, net, epoch, lr) -> None:
        self.net = net
        self.lr = lr
        self.epochs = epoch

    def acct(self, pred, label):
        p = torch.argmax(pred, 1).detach().numpy()
        label = label.detach().numpy()

        res = np.where(p == label)

        return len(res[0]) / len(label)

    def val(self, test_loader):
        self.net.eval()
        rv = 0
        for i, (data, label) in enumerate(test_loader):
            pred = self.net(data.float())
            rv += self.acct(pred, label)

        return rv / i

    def train(self, train_loader, test_loader):
        self.net.train()
        lm, am, vm = [], [], []
        cri = nn.CrossEntropyLoss()

        for e in range(1, self.epochs + 1):
            rl, ra = 0, 0
            if e % 5 == 0:
                self.lr = self.lr * 0.75
            opt = torch.optim.SGD(self.net.parameters(), lr=self.lr)
            for i, (data, label) in enumerate(train_loader):
                opt.zero_grad()
                pred = self.net(data.float())
                loss = cri(pred, label.long())
                rl += loss.item()
                ra += self.acct(pred, label.long())
                loss.backward()
                opt.step()
            lm.append(rl / i)
            am.append(ra / i)
            vm.append(self.val(test_loader))

            print(f"Current epoch: {e} Loss -> {rl/i} Acc ->{ra/i}")
        return lm, am, vm
