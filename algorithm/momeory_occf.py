import numpy as np
from algorithm.ranking_evaluation import data_precess, real_set, precision, recall


class OCCF:
    def __init__(self, data):
        self.m = 1682
        self.n = 943
        self.matrix = np.zeros((self.m, self.n))
        self.user_matrix = np.zeros((self.n, self.n))
        self.item_matrix = np.zeros((self.m, self.m))
        for u, i in data:
            self.matrix[i - 1][u - 1] = 1
        self.jac_index()
        # self.__confidence_matrix()
        # self.__normalization()

    def __confidence_matrix(self):
        for k in range(self.m):
            Uk = np.nonzero(self.matrix[k])[0]
            if Uk.size == 0:
                self.item_matrix[k] = 0
                continue
            for j in range(self.m):
                Uj = np.nonzero(self.matrix[j])[0]
                self.item_matrix[k][j] = np.intersect1d(Uk, Uj).size / Uk.size
        for w in range(self.n):
            Iw = np.nonzero(self.matrix[:, w])[0]
            if Iw.size == 0:
                self.user_matrix[w] = 0
                continue
            for u in range(self.n):
                Iu = np.nonzero(self.matrix[:, u])
                self.user_matrix[w][u] = np.intersect1d(Iw, Iu).size / Iw.size

    def jac_index(self):
        for k in range(self.m):
            Uk = np.nonzero(self.matrix[k])[0]
            if Uk.size == 0:
                self.item_matrix[k] = 0
                continue
            for j in range(self.m):
                Uj = np.nonzero(self.matrix[j])[0]
                if Uj.size != 0:
                    self.item_matrix[k][j] = np.intersect1d(Uk, Uj).size / (np.sqrt(Uk.size) * np.sqrt(Uj.size))

        for w in range(self.n):
            Iw = np.nonzero(self.matrix[:, w])[0]
            if Iw.size == 0:
                self.user_matrix[w] = 0
                continue
            for u in range(self.n):
                Iu = np.nonzero(self.matrix[:, u])[0]
                if Iu.size != 0:
                    self.user_matrix[w][u] = np.intersect1d(Iw, Iu).size / (np.sqrt(Iu.size) * np.sqrt(Iw.size))

    def __normalization(self):
        for u in range(self.n):
            frac = np.max(self.user_matrix[u])
            if frac != 0:
                self.user_matrix[u] = self.user_matrix[u] / frac

        for i in range(self.m):
            frac = np.max(self.item_matrix[i])
            if frac != 0:
                self.item_matrix[i] = self.item_matrix[i] / frac

    def nearest(self, method, K):
        if method == 'item_based':
            Nj = {}
            for j in range(self.m):
                idx = np.argsort(self.item_matrix[j])[::-1]
                Nj[j] = idx[:K]
            return Nj
        else:
            Nu = {}
            for u in range(self.n):
                idx = np.argsort(self.user_matrix[u])[::-1]
                Nu[u] = idx[:K]
            return Nu

    def item_based_occf(self, Nj):
        rating_matrix = np.zeros(self.matrix.shape)
        for u in range(self.n):
            Iu = np.nonzero(self.matrix[:, u])
            for j in range(self.m):
                kset = np.intersect1d(Iu, Nj[j])
                r = 0
                for k in kset:
                    r += self.item_matrix[k][j]
                rating_matrix[j][u] = r
        return rating_matrix

    def recommed(self, rating_matrix, k=5):
        re = {}
        for u in range(self.n):
            item = np.argsort(rating_matrix[:, u])[::-1]
            Iu = np.nonzero(self.matrix[:, u])
            for i in Iu:
                id = np.where(item == i)
                np.delete(item, id)
            re[u] = item[:k]
        return re

    def user_based_occf(self):
        pass


def main():
    trainfile = 'C:/Users/Jack/Desktop/Data/ml/u1.base.csv'
    testfile = 'C:/Users/Jack/Desktop/Data/ml/u1.test.csv'
    train_data = data_precess(trainfile)
    test_data = data_precess(testfile)
    occf = OCCF(train_data)
    Nj = occf.nearest('item_based', 50)
    ratings = occf.item_based_occf(Nj)
    test_users = np.unique(test_data[:, 0])
    Ite = real_set(test_data, test_users)
    Ire = occf.recommed(ratings)
    prec = precision(Ire, Ite, test_users, 5)
    print(prec[1])
    rec = recall(Ire, Ite, test_users, 5)
    print(rec[1])


if __name__ == '__main__':
    main()
