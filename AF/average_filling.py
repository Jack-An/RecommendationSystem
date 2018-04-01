import numpy as np
import warnings


class Average:
    """
    Average filling algorithm
    """

    def __init__(self, users, items, ratings):
        self.users = users
        self.items = items
        self.ratings = ratings
        self.matrix = np.zeros((np.max(users), np.max(items)))
        # fill the rating matrix
        # if the data is not observed,the position will be zero
        # index will be start with zero not 1
        # so use -1
        for i, j, k in zip(self.users, self.items, self.ratings):
            self.matrix[i - 1, j - 1] = k

    def global_average(self):
        """
        compute the global average of ratings (except not be observed)
        :return: global average
        """
        return self.matrix[self.matrix.nonzero()].mean()

    def user_average(self):
        """
        compute the average of users
        :return: a vector for each user
        """
        ru = []
        f = lambda t: t[t.nonzero()].mean()
        for user in self.matrix:
            ru.append(f(user))
        return np.array(ru)

    def item_average(self):
        """
        compute the average of item
        cause some item not be observed,
        so use with to ignore NAN problem,
        last,use global average to fill the NAN
        :return: a vector for each item
        """
        ri = []
        f = lambda t: t[t.nonzero()].mean()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for item in self.matrix.T:
                ri.append(f(item))
        ri = np.array(ri)
        ri[np.isnan(ri)] = self.global_average()
        return ri

    def bias(self, ru, ri):
        """
        compute bias of users and bias of items
        :param ru: user_average
        :param ri: item_average
        :return:a vector for each user & a vector for each item
        """
        bu = []
        for user in self.matrix:
            l = np.nonzero(user)
            user = user[l]
            r = ri[l]
            bu.append(np.mean(user - r))

        bi = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for item in self.matrix.T:
                l = np.nonzero(item)
                item = item[l]
                r = ru[l]
                bi.append(np.mean(item - r))
        return np.array(bu), np.nan_to_num(np.array(bi))

    @staticmethod
    def MAE(pred, y):
        """
        compute the MAE(mean absolutely error)
        :param pred: prediction
        :param y: real ratings
        :return:  MAE
        """
        return np.mean(np.abs(y - pred))

    @staticmethod
    def RMSE(pred, y):
        """
        compute RMSE(root mean square error)
        :param pred: prediction
        :param y: real ratings
        :return: RMSE
        """
        return np.sqrt(np.mean(np.square(y - pred)))

    @staticmethod
    def prediction(X, args1, method, args2=None, args3=None):
        """
        compute prediction for different method of predict rule
        :param X: input matrix
        :param args1:
        :param method: method type
        :param args2:
        :param args3:
        :return:
        """
        m, n = X.shape
        pred = np.zeros((m, 1), dtype=np.int)
        if method == 'Method 1':  # r = ru
            for i in range(m):
                pred[i, 0] = args1[X[i, 0] - 1]
        elif method == 'Method 2':  # r =ri
            for i in range(m):
                pred[i, 0] = args1[X[i, 1] - 1]
        elif method == 'Method 3':  # r = ru/2 + ri/2
            for i in range(m):
                pred[i, 0] = args1[X[i, 0] - 1] / 2 + args2[X[i, 0] - 1] / 2
        elif method == 'Method 4':  # r = bu + ri
            for i in range(m):
                pred[i, 0] = args1[X[i, 0] - 1] + args2[X[i, 1] - 1]
        elif method == 'Method 5':  # r = ru +bi
            for i in range(m):
                pred[i, 0] = args1[X[i, 0] - 1] + args2[X[i, 1] - 1]
        else:  # r = r(global average) + bu + bi
            for i in range(m):
                pred[i, 0] = args1 + args2[X[i, 0] - 1] + args3[X[i, 1] - 1]
        return pred


def main():
    # Load train data
    data = np.genfromtxt('C:\\Users\\Jack\\Desktop\\Data\\ml-100k\\ml-100k\\u1.base', delimiter='	', dtype=np.int)
    print(data.shape)
    users = data[:, 0]
    items = data[:, 1]
    ratings = data[:, 2]
    avg = Average(users, items, ratings)
    test = np.genfromtxt('C:\\Users\\Jack\\Desktop\\Data\\ml-100k\\ml-100k\\u1.test', delimiter='	', dtype=np.int)

    # Load test data
    X = test[:, :2]
    y = test[:, 2]

    # compute the parameters
    r = avg.global_average()
    ru = avg.user_average()
    ri = avg.item_average()
    bu, bi = avg.bias(avg.user_average(), avg.item_average())

    # training and compute accuracy
    pred = avg.prediction(X, args1=ru, method='Method 1')
    print('Method 1')
    print(avg.RMSE(pred, y))
    print(avg.MAE(pred, y))
    pred = avg.prediction(X, args1=ri, method='Method 2')
    print('Method 2')
    print(avg.RMSE(pred, y))
    print(avg.MAE(pred, y))
    pred = avg.prediction(X, args1=ru, args2=ri, method='Method 3')
    print('Method 3')
    print(avg.RMSE(pred, y))
    print(avg.MAE(pred, y))
    pred = avg.prediction(X, args1=bu, args2=ri, method='Method 4')
    print('Method 4')
    print(avg.RMSE(pred, y))
    print(avg.MAE(pred, y))
    pred = avg.prediction(X, args1=ru, args2=bi, method='Method 5')
    print('Method 5')
    print(avg.RMSE(pred, y))
    print(avg.MAE(pred, y))
    pred = avg.prediction(X, args1=r, args2=bu, args3=bi, method='Method 6')
    print('Method 6')
    print(avg.RMSE(pred, y))
    print(avg.MAE(pred, y))


if __name__ == '__main__':
    main()
