# main.py

import gzip
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error, classification_report
from PIL import Image

from dataset import *
from trainer import Trainer

lr = 1e-3
num_iter = 100
params = {
    'batch_size': 16,
    'shuffle': True,
    'num_workers': 1
}


def linear_regression():
    # get boston dataset
    boston_dataset = load_boston()

    # derive pandas.DataFrame format
    dataset = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
    dataset['MEDV'] = boston_dataset.target

    # get features and targets
    X = pd.DataFrame(np.c_[dataset['LSTAT'], dataset['RM']], columns=['LSTAT', 'RM'])
    Y = dataset['MEDV']

    # split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                        test_size=0.3,
                                                        random_state=5)

    # create train and test datasets and data_loaders
    train_dataset = HousingDataset(inputs=X_train.to_numpy(),
                                   labels=y_train.to_numpy())
    train_loader = DataLoader(train_dataset, **params)

    test_dataset = HousingDataset(inputs=X_test.to_numpy(),
                                  labels=y_test.to_numpy())
    test_loader = DataLoader(test_dataset, **params)

    # build model
    model = torch.nn.Sequential(
        torch.nn.Linear(X_train.shape[-1], 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 32),
        torch.nn.Tanh(),
        torch.nn.Linear(32, 1))

    # train model
    trainer = Trainer(optimizer=torch.optim.Adam(params=model.parameters(), lr=lr),
                      loss_fn=torch.nn.MSELoss())
    model = trainer.fit(model, train_loader, num_iter=num_iter)

    # evaluate
    train_predict = np.array(
        [model(x.float()).detach().numpy() for x in torch.tensor(X_train.to_numpy())])[:, 0]
    rmse = np.sqrt(mean_squared_error(y_train, train_predict))
    print('\nRMSE of the training set: {}'.format(rmse))
    test_predict = np.array(
        [model(x.float()).detach().numpy() for x in torch.tensor(X_test.to_numpy())])[:, 0]
    rmse = np.sqrt(mean_squared_error(y_test, test_predict))
    print('\nRMSE of the testing set: {}'.format(rmse))
    return None


def logistic_regression():
    def impute_age(cols):
        Age = cols[0]
        Pclass = cols[1]

        if pd.isnull(Age):

            if Pclass == 1:
                return 37

            elif Pclass == 2:
                return 29

            else:
                return 24

        else:
            return Age

    # read data
    train_ds = pd.read_csv('titanic/train.csv')
    test_ds = pd.read_csv('titanic/test.csv')

    # clean missing data in the 'Age' column
    train_ds['Age'] = train_ds[['Age', 'Pclass']].apply(impute_age, axis=1)
    test_ds['Age'] = test_ds[['Age', 'Pclass']].apply(impute_age, axis=1)

    # drop the ccabin and embarked columns
    train_ds.drop('Cabin', axis=1, inplace=True)
    test_ds.drop('Cabin', axis=1, inplace=True)
    train_ds.dropna(inplace=True)

    # convert categorical features
    sex = pd.get_dummies(train_ds['Sex'], drop_first=True)
    embark = pd.get_dummies(train_ds['Embarked'], drop_first=True)
    train_ds.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)
    train_ds = pd.concat([train_ds, sex, embark], axis=1)

    # split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(train_ds.drop('Survived', axis=1),
                                                        train_ds['Survived'],
                                                        test_size=0.2,
                                                        random_state=101)

    # create train and test datasets and data_loaders
    train_dataset = TitanicDataset(inputs=X_train.to_numpy(),
                                   labels=y_train.to_numpy())
    train_loader = DataLoader(train_dataset, **params)

    test_dataset = TitanicDataset(inputs=X_test.to_numpy(),
                                  labels=y_test.to_numpy())
    test_loader = DataLoader(test_dataset, **params)

    # build model
    model = torch.nn.Sequential(
        torch.nn.Linear(X_train.shape[-1], 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 3),
        torch.nn.Tanh(),
        torch.nn.Linear(3, 1)
    )

    # train model
    trainer = Trainer(optimizer=torch.optim.Adam(params=model.parameters(), lr=lr),
                      loss_fn=torch.nn.BCEWithLogitsLoss())
    model = trainer.fit(model, train_loader, num_iter=num_iter)

    # evaluate
    train_predict = np.array(
        [torch.sigmoid(model(x.float())).detach().numpy() for x in torch.tensor(X_train.to_numpy())])[:, 0]
    train_predict = [1 if x >= 0.5 else 0 for x in train_predict]
    print('\nThe classification performance on the training set \n{}'.format(
        classification_report(y_train, train_predict)))

    test_predict = np.array([torch.sigmoid(model(x.float())).detach().numpy() for x in torch.tensor(X_test.to_numpy())])
    test_predict = [1 if x >= 0.5 else 0 for x in test_predict]
    print(
        '\nThe classification performance on the testing set \n{}'.format(classification_report(y_test, test_predict)))
    return None


def fashion_classifier():
    # read data
    print('Loading data')
    with gzip.open('fashion-mnist/data/fashion/train-labels-idx1-ubyte.gz', 'rb') as file:
        train_labels = np.frombuffer(file.read(), dtype=np.uint8,
                                     offset=8)
    with gzip.open('fashion-mnist/data/fashion/train-images-idx3-ubyte.gz', 'rb') as file:
        train_images = np.frombuffer(file.read(), dtype=np.uint8,
                                     offset=16).reshape(len(train_labels), 784)
    with gzip.open('fashion-mnist/data/fashion/t10k-labels-idx1-ubyte.gz', 'rb') as file:
        test_labels = np.frombuffer(file.read(), dtype=np.uint8,
                                    offset=8)
    with gzip.open('fashion-mnist/data/fashion/t10k-images-idx3-ubyte.gz', 'rb') as file:
        test_images = np.frombuffer(file.read(), dtype=np.uint8,
                                    offset=16).reshape(len(test_labels), 784)

    # visualize first two images
    print('First two image in the training set')
    '''
    for i in range(2):
        img = train_images[i].reshape((28, 28))
        img = Image.fromarray(img)
        img.show()
    '''
    # create validation set
    X_train, X_val, y_train, y_val = train_test_split(train_images,
                                                      train_labels,
                                                      test_size=0.1,
                                                      random_state=101)

    # initialize datasets and loaders
    train_dataset = FashionDataset(images=X_train,
                                   labels=y_train)
    val_dataset = FashionDataset(images=X_val,
                                 labels=y_val)
    train_loader = DataLoader(train_dataset, **params)
    val_loader = DataLoader(val_dataset, **params)

    # build model
    cnn = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(3,3),
                        stride=(1,1), padding=(1,1)),
        torch.nn.BatchNorm2d(num_features=4),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2),
        torch.nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(3,3),
                        stride=(1,1), padding=(1,1)),
        torch.nn.BatchNorm2d(num_features=4),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=2, stride=2)
    )
    classifier = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(196, 10, bias=True),
        torch.nn.Softmax()
    )
    model = torch.nn.Sequential(cnn, classifier)

    # train model
    num_iter = 25
    trainer = Trainer(optimizer=torch.optim.Adam(params=model.parameters(), lr=lr),
                      loss_fn=torch.nn.CrossEntropyLoss())
    model = trainer.fit(model, train_loader, num_iter=num_iter,
                        val_loader=val_loader, acc_metric=True)
    pass


# execute regression and classification tasks
if __name__ == '__main__':
    # linear regression
    # linear_regression()

    # logistic regression
    logistic_regression()

    # q3:

    # q4
    #fashion_classifier()
