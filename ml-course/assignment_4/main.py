# import dependencies
import numpy as np
import pandas as pd
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNN

def generate_data(n_samples, tst_frac=0.2, val_frac=0.2):
    # Generate a non-linear data set
    X, y = make_moons(n_samples=n_samples, noise=0.25, random_state=42)

    # Take a small subset of the data and make it VERY noisy; that is, generate outliers
    m = 30
    np.random.seed(30) # Deliberately use a different seed
    ind = np.random.permutation(n_samples)[:m]
    X[ind, :] += np.random.multivariate_normal([0, 0], np.eye(2), (m, ))
    y[ind] = 1 - y[ind]

    # Plot this data
    cmap = ListedColormap(['#b30065', '#178000'])
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolors='k')

    # First, we use train_test_split to partition (X, y) into training and test sets
    X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=tst_frac,
                                                random_state=42)
    # Next, we use train_test_split to further partition (X_trn, y_trn) into tra ining and validation sets
    X_trn, X_val, y_trn, y_val = train_test_split(X_trn, y_trn, test_size=val_frac, random_state=42)
    return (X_trn, y_trn), (X_val, y_val), (X_tst, y_tst)

def main(args):
    # generate data
    n_samples = 300
    (X_trn, y_trn), (X_val, y_val), (X_tst, y_tst) = generate_data(n_samples)

    # effect of regularization parameter, C
    print('Effect of regularization parameter, C')
    C_range = np.arange(-3.0, 6.0, 1.0)
    C_values = np.power(10.0, C_range)
    models, trnErr, valErr = dict(), dict(), dict()
    for C in C_values:
        models[C] = SVC(C = C, kernel = 'rbf', gamma = 'scale')
        models[C].fit(X_trn, y_trn)
    max_score = 0
    best_model = None
    for C in C_values:
        # compute training and validation errors
        # training
        score = models[C].score(X_trn, y_trn)
        trnErr[str(models[C])] = score
        # validation
        score = models[C].score(X_val, y_val)
        valErr[str(models[C])] = score
        print("Score of {} = {}".format(models[C], score))
        if score > max_score:
            best_model = models[C]
            max_score = score
    print("The best model is {}".format(best_model))
    print("Accuracy of the best model {} is {}".format(best_model, best_model.score(X_tst, y_tst)))
    print()

    # effect of the RBF kernel parameter, y
    # Learn support vector classifiers with a radial-basis function kernel with # fixed C = 10.0 and different values of gamma
    print("Effect of the RBF kernal parameter, y")
    gamma_range = np.arange(-2.0, 4.0, 1.0)
    gamma_values = np.power(10.0, gamma_range)
    models = dict()
    trnErr = dict()
    valErr = dict()
    for G in gamma_values:
        models[G] = SVC(C = 10.0, kernel = 'rbf', gamma = G)
        models[G].fit(X_trn, y_trn)

    max_score = 0
    best_model = None
    for G in gamma_values:
        # compute training and validation errors
        # training
        score = models[G].score(X_trn, y_trn)
        trnErr[str(models[G])] = score
        # validation
        score = models[G].score(X_val, y_val)
        valErr[str(models[G])] = score
        print("Score of {} = {}".format(models[G], score))
        if score > max_score:
            best_model = models[G]
            max_score = score
    print("The best model is {}".format(best_model))
    print("Accuracy of the best model {} is {}".format(best_model, best_model.score(X_tst, y_tst)))
    print()

    # breast cancer diagnosis with SVM
    print("Breast Cancer Diagnosis with Support Vector Machine")
    # load data
    train_set = pd.read_csv('./wdbc_trn.csv', names = ['label'] + list(range(30)))
    val_set = pd.read_csv('./wdbc_val.csv', names = ['label'] + list(range(30)))
    tst_set = pd.read_csv('./wdbc_tst.csv', names = ['label'] + list(range(30)))
    # select model
    trnErr = []
    valErr = []
    models = dict()
    gamma_range = np.arange(-3.0, 3.0, 1.0)
    gamma_values = np.power(10.0, gamma_range)
    C_range = np.arange(-2.0, 5.0, 1.0)
    C_values = np.power(10.0, C_range)

    # training
    for C in C_values:
        trnErr.append([])
        for gamma in gamma_values:
            features, label = train_set[list(range(30))], train_set['label']
            svc = SVC(C = C, kernel = 'rbf', gamma = gamma)
            svc.fit(features, label)
            trnErr[-1].append(svc.score(features, label))
            models[str('C:{}_gamma:{}'.format(C, gamma))] = svc

    # validation
    max_score = 0
    best_model = None
    for C in C_values:
        valErr.append([])
        for gamma in gamma_values:
            features, label = val_set[list(range(30))], val_set['label']
            valErr[-1].append(models[str('C:{}_gamma:{}'.format(C, gamma))].score(features, label))
            if valErr[-1][-1] > max_score:
                max_scoree = valErr[-1]
                best_model = models[str('C:{}_gamma:{}'.format(C, gamma))]

    print("The best model is {}".format(best_model))

    # print training and validation errors
    # print training and validation errors
    trnErrArr = np.array(trnErr)
    valErrArr = np.array(valErr)

    # to dataframe
    trnErrArr = pd.DataFrame(data = trnErrArr,
                         columns = ['G={}'.format(x) for x in gamma_values],
                         index = ['C={}'.format(x) for x in C_values])

    valErrArr = pd.DataFrame(data = valErrArr,
                         columns = ['G={}'.format(x) for x in gamma_values],
                         index = ['C={}'.format(x) for x in C_values])
    print("Training Errors of SVM over various C and gamma values")
    print(trnErrArr)
    print()
    print("Validation Errors of SVM over various C and gamma values")
    print(valErrArr)

    features, label = tst_set[list(range(30))], tst_set['label']
    print("Test accuracy of the best model {} is {}".format(best_model, best_model.score(features, label)))
    print()

    # breast cancer diagnosis with k-nearest-neighbors
    k_values = [1, 5, 11, 15, 21]

    trnErr = []
    valErr = []
    models = dict()
    # training
    for k in k_values:
        features, label = train_set[list(range(30))], train_set['label']
        models[k] = KNN(n_neighbors = k, algorithm = 'kd_tree')
        models[k].fit(features, label)
        trnErr.append(models[k].score(features, label))
    # validation
    max_score = 0
    best_model = None
    best_k = 0
    for k in k_values:
        features, label = val_set[list(range(30))], val_set['label']
        valErr.append(models[k].score(features, label))
        if valErr[-1] > max_score:
            max_score = valErr[-1]
            best_model = models[k]
            best_k = k
    print("The best model is {} with k= {}".format(best_model, best_k))
    features, label = tst_set[list(range(30))], tst_set['label']
    print("Test accuracy of the best model {} with k = {} is {}".format(best_model, best_k, best_model.score(features, label)))

    return None

if __name__ == '__main__':
    main(None)
