# trainer.py
import torch

class Trainer(object):
    def __init__(self, optimizer, loss_fn, acc_metric=False):
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loss, self.val_loss = [], []
        self.train_acc, self.val_acc = [], []
        self.acc_metric=acc_metric

    def fit(self, model, data_loader, num_iter, val_loader=None):
        # Args
        #   - data: tuple(inputs, labels)
        #   - num_iter: ints
        #       Number of iterations

        step = 0
        # training pipeline
        for iter in range(num_iter):
            epoch_loss, epoch_acc, step = 0, 0, 0
            for inputs, labels in data_loader:
                # zero gradient
                self.optimizer.zero_grad()

                # make preds
                preds = model(inputs)

                # compute loss
                loss = self.loss_fn(preds, labels)

                if self.acc_metric:
                    # accuracy
                    epoch_acc += acc_fn(preds, labels)

                # backward
                loss.backward()
                self.optimizer.step()

                step += 1
                epoch_loss += loss.sum()


            self.train_loss.append(epoch_loss / step)
            self.train_acc.append(epoch_acc / step)
            if self.acc_metric:
                print('Iter: {}, Loss: {}, Acc: {}'.format(iter, self.train_loss[-1], self.train_acc[-1]))
            else:
                print('Iter: {}, Loss: {}'.format(iter, self.train_loss[-1]))

            if val_loader is not None:
                epoch_loss, step, epoch_acc = 0, 0, 0
                for inputs, labels in data_loader:
                    # make preds
                    preds = model(inputs)

                    # compute loss
                    loss = self.loss_fn(preds, labels)

                    # add loss
                    step += 1
                    epoch_loss += loss.sum()

                    if self.acc_metric:
                        # accuracy
                        epoch_acc += acc_fn(preds, labels)

                self.val_loss.append(epoch_loss / step)
                self.val_acc.append(epoch_acc / step)
                if self.acc_metric:
                    print('Iter: {}, Val_Loss: {}, Val_Acc: {}'.format(iter, self.val_loss[-1], self.val_acc[-1]))
                else:
                    print('Iter: {}, Val_Loss: {}'.format(iter, self.val_loss[-1]))
        return model

    def evaluate(self, model, data_loader):
        acc = []
        for inputs, labels in data_loader:
            # make preds
            preds = model(inputs)

            # accuracy
            acc.append(acc_fn(preds, labels))

        return torch.mean(torch.tensor(acc)) # average


def acc_fn(x, y):
    # select elements with the max prob
    x =  x.argmax(dim=-1)

    # compare w/ labels
    acc = (x == y).sum().float() / float(y.size(0))
    return acc