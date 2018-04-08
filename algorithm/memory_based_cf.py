import numpy as np
from algorithm.average_filling import Average
import scipy.stats as st
import scipy


class CF(Average):
    def __init__(self, users, items, ratings):
        super().__init__(users, items, ratings)
        self.user_simi = np.zeros((self.matrix.shape[0], self.matrix.shape[0]))
        self.item_simi = np.zeros((self.matrix.shape[1], self.matrix.shape[1]))

    def PCC(self):
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
                # if user1 == user2:
                #     self.simi[user1][user2] = 1

    def ACC(self):
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
        print('Starting prediction(UCF) ...')
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
        print('Ending prediction(UCF) ...')
        return pred

    def predict_item_based(self, X, Nju):
        print('Starting prediction(ICF) ...')
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
        print('Ending prediction(ICF) ...')
        return pred

    @staticmethod
    def predict_hybrid_cf(ucf, icf, lambd):
        print('Starting prediction(HCF) ...')
        pred = lambd * ucf + (1 - lambd) * icf
        print('Ending prediction(HCF) ...')
        return pred


def main():
    data = np.genfromtxt('C:\\Users\\Jack\\Desktop\\Data\\ml-100k\\ml-100k\\u1.base', delimiter='	', dtype=np.int)
    users = data[:, 0]
    items = data[:, 1]
    ratings = data[:, 2]
    cf = CF(users, items, ratings)

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
    print('RMSE : {:f}'.format(CF.RMSE(pred_ucf, y)))
    print('MAE  : {:f}'.format(CF.MAE(pred_ucf, y)))

    # Item based CF
    cf.ACC()
    Nju = cf.item_based(50)
    pred_icf = cf.predict_item_based(X, Nju)
    print('Item based CF ')
    print('RMSE : {:f}'.format(CF.RMSE(pred_icf, y)))
    print('MAE  : {:f}'.format(CF.MAE(pred_icf, y)))

    # Hybrid CF
    pred_hybrid = CF.predict_hybrid_cf(ucf=pred_ucf, icf=pred_icf, lambd=0.5)
    print('Item based CF ')
    print('RMSE : {:f}'.format(CF.RMSE(pred_hybrid, y)))
    print('MAE  : {:f}'.format(CF.MAE(pred_hybrid, y)))


if __name__ == '__main__':
    main()
