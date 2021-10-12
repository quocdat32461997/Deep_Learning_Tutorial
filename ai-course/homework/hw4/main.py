# main.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDRegressor
from sklearn.metrics import mean_squared_error, classification_report

from optimizers import *
from losses import *
from trainer import Trainer
from models import LogisticRegressor

def linear_regression():
    # get boston dataset
    boston_dataset = load_boston()

    # derive pandas.DataFrame format
    dataset = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
    dataset['MEDV'] = boston_dataset.target

    # get features and targets
    X = pd.DataFrame(np.c_[dataset['LSTAT'], dataset['RM']], columns = ['LSTAT','RM'])
    Y = dataset['MEDV']

    # split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                        test_size=0.3,
                                                        random_state=5)

    # Q1: Linear Regression w/o SGD
    print('Q1: Training Linear Regression models on the Boston Housing dataset without SGD.')
    # create linear_regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # evaluate
    y_train_predict = model.predict(X_train)
    rmse = np.sqrt(mean_squared_error(y_train, y_train_predict))
    print('\nRMSE of the training set: {}'.format(rmse))
    y_test_predict = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_predict))
    print('\nRMSE of the testing set: {}'.format(rmse))

    # Q2.1: Linear Regression w/ Gradient Descent
    # initialize hyper-parameters
    num_iter = 50
    lr = 1e-7

    print('\n Q2.1: Training Linear Regression models on the Boston Housing dataset with GD.')
    sgd_trainer = Trainer(optimizer=GradientDescent(model=model, loss_fn=MSE, learning_rate=lr))
    sgd_trainer.train(data=(X_train.to_numpy(), y_train.to_numpy()),
                      num_iter=num_iter)
    _model = sgd_trainer.get_model()
    # evaluate
    y_train_predict = _model.predict(X_train)
    rmse = np.sqrt(mean_squared_error(y_train, y_train_predict))
    print('\nRMSE of the training set: {}'.format(rmse))
    y_test_predict = _model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_predict))
    print('\nRMSE of the testing set: {}'.format(rmse))

    # Q2.2: Linear Regression w/ SGD
    lr = 3e-5
    print('\nQ2.2: Training Linear Regression models on the Boston Housing dataset with SGD.')
    sgd_trainer = Trainer(optimizer=SGD(model=model,loss_fn=MSE, learning_rate=lr))
    sgd_trainer.train(data=(X_train.to_numpy(), y_train.to_numpy()),
                      num_iter=num_iter)
    _model = sgd_trainer.get_model()
    # evaluate
    y_train_predict = _model.predict(X_train)
    rmse = np.sqrt(mean_squared_error(y_train, y_train_predict))
    print('\nRMSE of the training set: {}'.format(rmse))
    y_test_predict = _model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_predict))
    print('\nRMSE of the testing set: {}'.format(rmse))

    # Q2.3: Linear Regression w/ SGD-momentums
    momentum=0.9

    print('\nQ2.3: Training Linear Regression models on the Boston Housing dataset with SGD w/ Momentum.')
    sgd_m_trainer = Trainer(optimizer=SGD(model=model,
                                        loss_fn=MSE,
                                        learning_rate=lr,
                                        momentum=momentum))
    sgd_m_trainer.train(data=(X_train.to_numpy(), y_train.to_numpy()),
                      num_iter=num_iter)
    _model = sgd_m_trainer.get_model()
    # evaluate
    y_train_predict = _model.predict(X_train)
    rmse = np.sqrt(mean_squared_error(y_train, y_train_predict))
    print('\nRMSE of the training set: {}'.format(rmse))
    y_test_predict = _model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_predict))
    print('\nRMSE of the testing set: {}'.format(rmse))

    # Q2.4: Linear Regression w/ SGD Nesterov-momentum
    print('\nQ2.4: Training Linear Regression models on the Boston Housing dataset with SGD w/ Nesterov Momentum.')
    sgd_m_trainer = Trainer(optimizer=SGD(model=model,
                                          loss_fn=MSE,
                                          learning_rate=lr,
                                          momentum=momentum,
                                          nesterov=True))
    sgd_m_trainer.train(data=(X_train.to_numpy(), y_train.to_numpy()),
                        num_iter=num_iter)
    _model = sgd_m_trainer.get_model()
    # evaluate
    y_train_predict = _model.predict(X_train)
    rmse = np.sqrt(mean_squared_error(y_train, y_train_predict))
    print('\nRMSE of the training set: {}'.format(rmse))
    y_test_predict = _model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_predict))
    print('\nRMSE of the testing set: {}'.format(rmse))

    # Q2.5: Linear Regression w/ AdaGradSGD
    lr = 1e-2
    num_iter = 100

    print('\nQ2.5: Training Linear Regression models on the Boston Housing dataset with AdaGradSGD.')
    sgd_m_trainer = Trainer(optimizer=AdaGrad(model=model,
                                          loss_fn=MSE,
                                          learning_rate=lr))
    sgd_m_trainer.train(data=(X_train.to_numpy(), y_train.to_numpy()),
                        num_iter=num_iter)
    _model = sgd_m_trainer.get_model()
    # evaluate
    y_train_predict = _model.predict(X_train)
    rmse = np.sqrt(mean_squared_error(y_train, y_train_predict))
    print('\nRMSE of the training set: {}'.format(rmse))
    y_test_predict = _model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_predict))
    print('\nRMSE of the testing set: {}'.format(rmse))

    # Q5.1: Linear Regression w/ Adam
    # initialize hyper-parameters
    lr = 1e-2
    num_iter = 50

    print('\nQ5.1: Training Linear Regression models on the Boston Housing dataset with Adam.')
    sgd_m_trainer = Trainer(optimizer=Adam(model=model,
                                              loss_fn=MSE,
                                              learning_rate=lr))
    sgd_m_trainer.train(data=(X_train.to_numpy(), y_train.to_numpy()),
                        num_iter=num_iter)
    _model = sgd_m_trainer.get_model()
    # evaluate
    y_train_predict = _model.predict(X_train)
    rmse = np.sqrt(mean_squared_error(y_train, y_train_predict))
    print('\nRMSE of the training set: {}'.format(rmse))
    y_test_predict = _model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_predict))
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

    # Q3: Logistic Regression w/o SGDs
    print('\nQ3: Training Linear Regression models on the Boston Housing dataset without SGD.')
    # creete model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # evaluate
    train_predict = model.predict(X_train)
    print('\nThe classification performance on the training set \n{}'.format(classification_report(y_train, train_predict)))

    test_predict = model.predict(X_test)
    print('\nThe classification performance on the testing set \n{}'.format(classification_report(y_test, test_predict)))

    # Q4.1: Logistic Regression w/ Gradient Descent
    # create model
    model = LogisticRegressor()
    model.fit(X_train, y_train)

    # parameters
    lr = 1e-3
    num_iter = 50

    print('\nQ4.1: Training Logistic Regression models on the Titanic dataset with Gradient Descent.')
    sgd_trainer = Trainer(optimizer=GradientDescent(model=model, loss_fn=CrossEntropy, learning_rate=lr))
    sgd_trainer.train(data=(X_train.to_numpy(), y_train.to_numpy()),
                      num_iter=num_iter)
    _model = sgd_trainer.get_model()

    # evaluate
    train_predict = _model.binary_predict(X_train)
    print('\nThe classification performance on the training set \n{}'.format(classification_report(y_train, train_predict)))

    test_predict = _model.binary_predict(X_test)
    print('\nThe classification performance on the testing set \n{}'.format(classification_report(y_test, test_predict)))

    # Q4.2: Logistic Regression w/ SGD
    lr = 3e-5
    print('\nQ4.2: Training Logistic Regression models on the Titanic dataset with SGDs.')
    sgd_trainer = Trainer(optimizer=SGD(model=model, loss_fn=CrossEntropy, learning_rate=lr))
    sgd_trainer.train(data=(X_train.to_numpy(), y_train.to_numpy()),
                      num_iter=num_iter)
    _model = sgd_trainer.get_model()

    # evaluate
    train_predict = _model.binary_predict(X_train)
    print('\nThe classification performance on the training set \n{}'.format(classification_report(y_train, train_predict)))

    test_predict = _model.binary_predict(X_test)
    print('\nThe classification performance on the testing set \n{}'.format(classification_report(y_test, test_predict)))

    # Q4.3: Logistic Regression w/ SGD momentum
    print('\nQ4.3: Training Logistic Regression models on the Titanic dataset with SGD momentum.')
    momentum = 0.9

    sgd_m_trainer = Trainer(optimizer=SGD(model=model,
                                          loss_fn=MSE,
                                          learning_rate=lr,
                                          momentum=momentum))
    sgd_m_trainer.train(data=(X_train.to_numpy(), y_train.to_numpy()),
                        num_iter=num_iter)
    _model = sgd_m_trainer.get_model()

    # evaluate
    train_predict = _model.binary_predict(X_train)
    print('\nThe classification performance on the training set \n{}'.format(classification_report(y_train, train_predict)))

    test_predict = _model.binary_predict(X_test)
    print('\nThe classification performance on the testing set \n{}'.format(classification_report(y_test, test_predict)))


    # Q4.3: Logistic Regression w/ SGD Nesterov momentum
    print('\nQ4.3: Training Logistic Regression models on the Titanic dataset with SGD w/ Nesterov momentum.')
    sgd_m_trainer = Trainer(optimizer=SGD(model=model,
                                          loss_fn=MSE,
                                          learning_rate=lr,
                                          momentum=momentum,
                                          nesterov=True))
    sgd_m_trainer.train(data=(X_train.to_numpy(), y_train.to_numpy()),
                        num_iter=num_iter)
    _model = sgd_m_trainer.get_model()

    # evaluate
    train_predict = _model.binary_predict(X_train)
    print('\nThe classification performance on the training set \n{}'.format(classification_report(y_train, train_predict)))

    test_predict = _model.binary_predict(X_test)
    print('\nThe classification performance on the testing set \n{}'.format(classification_report(y_test, test_predict)))

    # Q4.4: Logistic Regression w/ AdaGrad
    lr = 1e-2
    num_iter = 100

    print('\nQ4.4: Training Logistic Regression models on the Titanic dataset with AdaGrad.')
    sgd_m_trainer = Trainer(optimizer=Adam(model=model,
                                          loss_fn=MSE,
                                          learning_rate=lr))
    sgd_m_trainer.train(data=(X_train.to_numpy(), y_train.to_numpy()),
                        num_iter=num_iter)
    _model = sgd_m_trainer.get_model()

    # evaluate
    train_predict = _model.binary_predict(X_train)
    print('\nThe classification performance on the training set \n{}'.format(classification_report(y_train, train_predict)))

    test_predict = _model.binary_predict(X_test)
    print('\nThe classification performance on the testing set \n{}'.format(classification_report(y_test, test_predict)))

    # Q4.5: Logistic Regression w/ Adam
    lr = 1e-2
    num_iter = 50
    print('\nQ4.5: Training Logistic Regression models on the Titanic dataset with Adam.')
    sgd_m_trainer = Trainer(optimizer=AdaGrad(model=model,
                                              loss_fn=MSE,
                                              learning_rate=lr))
    sgd_m_trainer.train(data=(X_train.to_numpy(), y_train.to_numpy()),
                        num_iter=num_iter)
    _model = sgd_m_trainer.get_model()

    # evaluate
    train_predict = _model.binary_predict(X_train)
    print('\nThe classification performance on the training set \n{}'.format(classification_report(y_train, train_predict)))

    test_predict = _model.binary_predict(X_test)
    print('\nThe classification performance on the testing set \n{}'.format(classification_report(y_test, test_predict)))
    return None


# execute regression and classification tasks
if __name__ == '__main__':
    # linear regression
    linear_regression()

    # logistic regression
    logistic_regression()
