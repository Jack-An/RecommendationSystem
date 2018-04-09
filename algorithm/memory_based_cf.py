import numpy as np
from algorithm.average_filling import Average
import scipy.stats
import matplotlib.pyplot as plt
import pandas as pd


class CollaborativeFiltering(Average):
    """
    This file include user based CF ,item based Cf
    and Hybrid CF
    This class based on class Average
    """

    def __init__(self, users, items, ratings):
        super().__init__(users, items, ratings)
        self.user_simi = np.zeros((self.matrix.shape[0], self.matrix.shape[0]))
        self.item_simi = np.zeros((self.matrix.shape[1], self.matrix.shape[1]))

    def PCC(self):
        """
        compute the pearson correlation coefficient
        for each pair of users
        :return: None
        """
        m, n = self.matrix.shape
        for user1 in range(m):
            for user2 in range(m):
                if np.count_nonzero(self.matrix[user1]) and np.count_nonzero(self.matrix[user2]):
                    try:
                        p = scipy.stats.pearsonr(self.matrix[user1], self.matrix[user2])[0]
                        if not np.isnan(p):
                            self.user_simi[user1][user2] = p
                        else:
                            self.user_simi[user1][user2] = 0
                    except:
                        self.user_simi[user1][user2] = 0

    def ACC(self):
        """
        compute the pearson correlation coefficient
        for each pair of items
        :return: None
        """
        m, n = self.matrix.shape
        for item1 in range(n):
            for item2 in range(n):
                if np.count_nonzero(self.matrix.T[item1]) and np.count_nonzero(self.matrix.T[item2]):
                    try:
                        p = scipy.stats.pearsonr(self.matrix.T[item1], self.matrix.T[item2])[0]
                        if not np.isnan(p):
                            self.item_simi[item1][item2] = p
                        else:
                            self.item_simi[item1][item2] = 0
                    except:
                        self.item_simi[item1][item2] = 0

    def user_based(self, K):
        """
        obtain the neighbors of each user where swu !=0 Nu
        Obtain the users who rated item j Uj
        Obtain a set of Top-k nearest neighbors Nuj
        :param K: Top-k
        :return: Nuj (a matrix)
        """
        m, n = self.matrix.shape
        Nu = []
        for user in self.user_simi:
            Nu.append(np.where(user > 0))

        Uj = []
        for item in self.matrix.T:
            Uj.append(np.nonzero(item))

        Nuj = []
        for u in range(m):
            for j in range(n):
                inters = np.intersect1d(Nu[u], Uj[j])
                if inters.size <= K:
                    Nuj.append(inters)
                else:
                    Nuj.append(self.__find_top_k_user(u, inters, K))
        return Nuj

    def item_based(self, K):
        """
        same to user based
        :param K:
        :return: Nju
        """
        m, n = self.matrix.shape
        Nj = []
        for item in self.item_simi:
            Nj.append(np.nonzero(item))

        Iu = []
        for user in self.matrix:
            Iu.append(np.nonzero(user))
        Nju = []
        for j in range(n):
            for u in range(m):
                inters = np.intersect1d(Nj[j], Iu[u])
                if inters.size <= K:
                    Nju.append(inters)
                else:
                    Nju.append(self.__find_top_k_item(j, inters, K))
        return Nju

    def __find_top_k_item(self, i, inters, K):
        """
        help find top-K
        :param i:
        :param inters:
        :param K:
        :return:
        """
        diction = dict()
        simi = self.item_simi[i]
        for item in inters:
            diction[item] = simi[item]
        sort_simi = sorted(diction.items(), key=lambda x: x[1], reverse=True)
        nei = []
        for i in range(K):
            nei.append(sort_simi[i][0])
        return nei

    def __find_top_k_user(self, u, inters, K):
        diction = dict()
        simi = self.user_simi[u]
        for user in inters:
            diction[user] = simi[user]
        sort_simi = sorted(diction.items(), key=lambda x: x[1], reverse=True)
        nei = []
        for i in range(K):
            nei.append(sort_simi[i][0])
        return nei

    def predict_user_based(self, X, Nuj):
        """
        user based predict rule
        :param X: input matrix
        :param Nuj: Nuj
        :return: pred
        """
        m = X.shape[0]
        ru = self.user_average()
        Nuj = np.array(Nuj).reshape(self.matrix.shape)
        pred = []
        for i in range(m):
            user_id = X[i][0] - 1
            item_id = X[i][1] - 1
            w = Nuj[user_id][item_id]
            if len(w) != 0:
                swu = self.user_simi[user_id][w]
                rwj = self.matrix.T[item_id][w]
                rw = ru[w]
                part = np.sum(swu * (rwj - rw)) / np.sum(swu)
                pred.append(ru[user_id] + part)
            else:
                pred.append(ru[user_id])
        pred = np.array(pred).flatten()
        for i in range(pred.size):
            if pred[i] > 5:
                pred[i] = 5
            if pred[i] < 1:
                pred[i] = 1
        return pred

    def predict_item_based(self, X, Nju):
        """
        item based predict rule
        :param X: input matrix
        :param Nju: Nju
        :return: pred
        """
        m = X.shape[0]
        ru = self.user_average()
        Nju = np.array(Nju).reshape(self.matrix.shape[1], self.matrix.shape[0])
        pred = []
        for i in range(m):
            user_id = X[i][0] - 1
            item_id = X[i][1] - 1
            w = Nju[item_id][user_id]
            if len(w) != 0:
                skj = self.item_simi[item_id][w]
                ruk = self.matrix[user_id][w]
                pred.append(np.sum(skj * ruk) / np.sum(skj))
            else:
                pred.append(ru[user_id])
        pred = np.array(pred).flatten()
        for i in range(pred.size):
            if pred[i] > 5:
                pred[i] = 5
            if pred[i] < 1:
                pred[i] = 1
        return pred

    @staticmethod
    def predict_hybrid_cf(ucf, icf, lambd):
        """
        Hybrid predict rule
        :param ucf: user based pred vector
        :param icf: item based pred vector
        :param lambd: lambda
        :return: pred
        """
        pred = lambd * ucf + (1 - lambd) * icf
        return pred


def main():
    data = np.genfromtxt('C:\\Users\\Jack\\Desktop\\Data\\ml-100k\\ml-100k\\u1.base', delimiter='	', dtype=np.int)
    users = data[:, 0]
    items = data[:, 1]
    ratings = data[:, 2]
    cf = CollaborativeFiltering(users, items, ratings)

    # User based CF
    # Load test data
    test = np.genfromtxt('C:\\Users\\Jack\\Desktop\\Data\\ml-100k\\ml-100k\\u1.test', delimiter='	', dtype=np.int)
    X = test[:, :2]
    y = test[:, 2]

    # User based CF
    cf.PCC()
    Nuj = cf.user_based(50)
    pred_ucf = cf.predict_user_based(X, Nuj)
    print('User based CF ')
    rmse1 = CollaborativeFiltering.RMSE(pred_ucf, y)
    mae1 = CollaborativeFiltering.MAE(pred_ucf, y)
    print('RMSE : {:f}'.format(rmse1))
    print('MAE  : {:f}'.format(mae1))

    # Item based CF
    cf.ACC()
    Nju = cf.item_based(50)
    pred_icf = cf.predict_item_based(X, Nju)
    print('Item based CF ')
    rmse2 = CollaborativeFiltering.RMSE(pred_icf, y)
    mae2 = CollaborativeFiltering.MAE(pred_icf, y)
    print('RMSE : {:f}'.format(rmse2))
    print('MAE  : {:f}'.format(mae2))

    # Hybrid CF
    pred_hybrid = CollaborativeFiltering.predict_hybrid_cf(ucf=pred_ucf, icf=pred_icf, lambd=0.5)
    print('Hybrid CF ')
    rmse3 = CollaborativeFiltering.RMSE(pred_hybrid, y)
    mae3 = CollaborativeFiltering.MAE(pred_hybrid, y)
    print('RMSE : {:f}'.format(rmse3))
    print('MAE  : {:f}'.format(mae3))
    data = np.array([rmse1, mae1, rmse2, mae2, rmse3, mae3]).reshape(3, 2)
    labels = ('User based CF', 'Item based CF', 'Hybrid CF')
    CollaborativeFiltering.plot(data, labels)


if __name__ == '__main__':
    main()
