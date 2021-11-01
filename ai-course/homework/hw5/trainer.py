# trainer.py


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
            epoch_loss = 0
            for inputs, labels in data_loader:
                # zero gradient
                self.optimizer.zero_grad()

                # make preds
                preds = model(inputs)

                # compute loss
                loss = self.loss_fn(preds, labels)

                if self.acc_metric:
                    # accuracy
                    self.train_acc.append(self.acc_fn(preds, labels))

                # backward
                loss.backward()
                self.optimizer.step()

                step += 1
                epoch_loss += loss.sum()
                if step % 100 == 0:
                    if self.acc_metric:
                        print('Step: {}, Loss: {}, Acc: {}'.format(step, epoch_loss / 100, sum(self.train_acc[-100:]) / 100))
                    else:
                        print('Step: {}, Loss: {}'.format(step, epoch_loss / 100))
                    self.train_loss.append(epoch_loss / 100)
                    epoch_loss = 0

            if val_loader is not None:
                epoch_loss, step = 0, 0
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
                        self.val_acc.append(self.acc_fn(preds, labels))
                if self.acc_metric:
                    print('Step: {}, Val_Loss: {}, Val_Acc: {}'.format(step, epoch_loss / step, sum(self.val_acc) / step))
                else:
                    print('Step: {}, Val_Loss: {}'.format(step, epoch_loss / step))
                self.val_loss.append(epoch_loss / step)
        return model


def acc_fn(x, y):
    # select elements with the max prob
    x =  x.argmax(dim=-1)

    # compare w/ labels
    acc = (x == y).sum().float() / float(y.size(0))
    return acc#.item()