import numpy as np
from algorithm.average_filling import Average
from algorithm.matrix_factorization import MatrixFactor


class SVDPP(Average):
    def __init__(self, users, items, ratings, implicit):
        super().__init__(users, items, ratings)
        self.I = []
        m = np.max(users)
        for i in range(m):
            user_set = []
            for j in range(implicit.shape[0]):
                if implicit[j, 0] == i + 1:
                    user_set.append(implicit[j, 1] - 1)
            self.I.append(user_set)

    @staticmethod
    def gradient(Uu, Vi, Iu, W, bu, bi, mu, x, alpha_u, alpha_v, alpha_w, beta_u, beta_v):
        assert W.shape[0] == 1682
        Wi = W[Iu]
        part = np.sum(Wi, axis=0) / np.sqrt(np.size(Iu))
        pred = Uu.dot(Vi.T) + part.dot(Vi.T) + bu + bi + mu
        e = pred - x[2]
        item = x[1]
        grad_Uu = e * Vi + alpha_u * Uu
        grad_Vi = e * (Uu + part) + alpha_v * Vi
        grad_bu = e + beta_u * bu
        grad_bi = e + beta_v * bi
        grad_mu = e
        grad_Wi = e * Vi / np.sqrt(np.size(Iu)) + alpha_w * W[item - 1]

        return grad_Uu, grad_Vi, grad_Wi, grad_bu, grad_bi, grad_mu

    def sgd_svd_pp(self, T, X, mu, bu, bi, alpha_u, alpha_v, alpha_w, beta_u, beta_v, gamma):
        d = 20
        n = 943
        m = 1682
        U = (np.random.rand(n, d) - 0.5) * 0.01
        V = (np.random.rand(m, d) - 0.5) * 0.01
        W = (np.random.rand(m, d) - 0.5) * 0.01
        for t in range(T):
            for j in range(X.shape[0]):
                x = X[j]
                u, i, r = x
                grad_Uu, grad_Vi, grad_Wi, grad_bu, grad_bi, grad_mu = SVDPP.gradient(U[u - 1], V[i - 1],
                                                                                      self.I[u - 1], W, bu[u - 1],
                                                                                      bi[i - 1], mu,
                                                                                      x, alpha_u, alpha_v, alpha_w,
                                                                                      beta_u, beta_v)
                U[u - 1] -= gamma * grad_Uu
                V[i - 1] -= gamma * grad_Vi
                bu[u - 1] -= gamma * grad_bu
                bi[i - 1] -= gamma * grad_bi
                W[i - 1] -= gamma * grad_Wi
                mu -= gamma * grad_mu
            gamma = 0.9 * gamma
        return U, V, W, bu, bi, mu

    def predict_svd_pp(self, X, U, V, W, bu, bi, mu):
        pred = []
        for u, i in X:
            Iu = self.I[u - 1]
            Wi = W[Iu]
            part = np.sum(Wi, axis=0) / np.sqrt(np.size(Iu))
            pred.append(U[u - 1].dot(V[i - 1].T) + part.dot(V[i - 1].T) + bu[u - 1] + bi[i - 1] + mu)
        for i in range(len(pred)):
            if pred[i] > 5:
                pred[i] = 5
            if pred[i] < 1:
                pred[i] = 1
        return np.array(pred).flatten()


def main():
    data = np.genfromtxt('C:\\Users\\Jack\\Desktop\\Data\\ml-100k\\ml-100k\\ua.base', delimiter='	', dtype=np.int)
    X_orig = data[:, :3]
    X = X_orig[np.random.permutation(X_orig.shape[0])]
    implicit = X[:int(X.shape[0] / 2), :]
    explicit = X[int(X.shape[0] / 2):, :]
    svd = SVDPP(X_orig[:, 0], X_orig[:, 1], X_orig[:, 2], implicit)
    mu = svd.global_average()
    ru = svd.user_average()
    ri = svd.item_average()
    bu, bi = svd.bias(ru, ri)
    alpha_u = alpha_v = alpha_w = beta_u = beta_v = gamma = 0.01
    iterations = 10
    U, V, W, bu, bi, mu = svd.sgd_svd_pp(iterations, explicit, mu, bu, bi, alpha_u, alpha_v, alpha_w, beta_u, beta_v,
                                         gamma)

    test = np.genfromtxt('C:\\Users\\Jack\\Desktop\\Data\\ml-100k\\ml-100k\\ua.test', delimiter='	', dtype=np.int)
    X = test[:, :2]
    y = test[:, 2]
    pred = svd.predict_svd_pp(X, U, V, W, bu, bi, mu)
    rmse1 = svd.RMSE(pred, y)
    mae1 = svd.MAE(pred, y)
    print('SVD++')
    print('RMSE : {:f}'.format(rmse1))
    print('MAE  :{:f}'.format(mae1))
    iterations = 20
    U, V, bu, bi, mu = MatrixFactor.rsvd(iterations, data[:, :3], mu, bu, bi,
                                         alpha_u, alpha_v,
                                         beta_u, beta_v,
                                         gamma)
    pred = MatrixFactor.rsvd_predict(X, U, V, bu, bi, mu)
    print('RSVD:')
    rmse2 = MatrixFactor.RMSE(pred, y)
    mae2 = MatrixFactor.MAE(pred, y)
    print('RMSE : {:f}'.format(rmse2))
    print('MAE  : {:f}'.format(mae2))
    plot_data = np.array([rmse1, mae1, rmse2, mae2]).reshape(2, 2)
    labels = ('SVD++', 'RSVD')
    svd.plot(plot_data, labels)


if __name__ == '__main__':
    main()
