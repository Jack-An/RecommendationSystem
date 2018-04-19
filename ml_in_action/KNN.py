import numpy as np
import matplotlib.pyplot as plt
import os


def load_data(filename):
    data = np.genfromtxt(filename, delimiter='	')
    p = np.random.permutation(data.shape[0])
    data = data[p]
    X = data[:, :3]
    y = np.array(data[:, 3], dtype=np.int)
    return X, y


def kNN(X_train, X_test, y_train, k):
    pred = []
    for t in X_test:
        # calculate the distance with L2 norm
        dis = np.linalg.norm(X_train - t, axis=1)
        # get k nearest neighbor
        nn = np.argsort(dis)[:k]
        nn_labels = y_train[nn]
        # get neighbor label which is most frequency
        pred.append(np.bincount(nn_labels).argmax())
    return pred


def normalization(X):
    # normalization the input matrix X if necessary
    # mu and sigma will be used for new input test set
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X = (X - mu) / sigma
    return X, mu, sigma


def plot(X, y):
    plt.scatter(X[:, 1], X[:, 2], 15.0 * y, 15.0 * y)
    plt.show()


def accuracy(pred, y_test):
    return np.where((np.array(pred) - y_test) == 0)[0].size / np.size(y_test)


def img2vector(filename):
    x = []
    with open(filename) as f:
        for line in f:
            x.extend(line.strip('\n'))
    return np.array(x, dtype=np.int)


def load_digits(dirpath):
    files = os.listdir(dirpath)
    X = []
    y = []
    for file in files:
        X.append(img2vector(dirpath + file))
        y.append(int(file.split('_')[0]))
    return np.array(X), np.array(y)


def classify_person():
    filename = 'C:/Users/Jack/Desktop/Data/ml_in_action/datingTestSet2.txt'
    X, y = load_data(filename)
    X, mu, sigma = normalization(X)
    split = int(X.shape[0] * 0.7)
    # train test(similar)
    X_train = X[:split]
    y_train = y[:split]
    # test set
    X_test = X[split:]
    y_test = y[split:]
    pred = kNN(X_train, X_test, y_train, 3)
    print('classify person accuracy : {:.4%}'.format(accuracy(pred, y_test)))


def classify_digits():
    # load data
    training_dir_path = 'C:/Users/Jack/Desktop/Data/ml_in_action/digits/trainingDigits/'
    test_dir_path = 'C:/Users/Jack/Desktop/Data/ml_in_action/digits/testDigits/'
    X_train, y_train = load_digits(training_dir_path)
    X_test, y_test = load_digits(test_dir_path)
    # make data set random
    p1 = np.random.permutation(y_train.size)
    X_train, y_train = X_train[p1], y_train[p1]
    p2 = np.random.permutation(y_test.size)
    X_test, y_test = X_test[p2], y_test[p2]

    pred = kNN(X_train, X_test, y_train, 3)
    print('classify digits accuracy : {:.4%}'.format(accuracy(pred, y_test)))


def main():
    classify_person()
    classify_digits()


if __name__ == '__main__':
    main()
