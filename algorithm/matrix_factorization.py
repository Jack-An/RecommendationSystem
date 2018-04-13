import numpy as np
from algorithm.average_filling import Average


class MatrixFactor(Average):
    """
    This filw will implement the Pure SVD, PMF, RSVD method in
    Recommendation system.
    This class based on Average, cause SVD mathod will use
    the rating matrix.
    Other method will not use any method or property in Average,
    so the other method will be static. (except evaluation and plot method)
    """

    def __init__(self, user, item, ratings):
        super().__init__(user, item, ratings)

    def PSVD(self):
        """
        PSVD method use Singular value decomposition to predict ratings
        :return: prediction rating's matrix
        """
        avg = self.user_average()
        for i in range(self.matrix.shape[0]):
            self.matrix[i][np.argwhere(self.matrix[i] == 0).flatten()] = avg[i]
        R = self.matrix - avg.reshape(self.matrix.shape[0], 1) * np.ones(self.matrix.shape)
        U, sigma, V_T = np.linalg.svd(R, full_matrices=False)
        U = U[:, :20]
        Sigma = np.diag(sigma[:20])
        V = V_T[:20, :]
        R_ = U.dot(Sigma).dot(V) + avg.reshape(self.matrix.shape[0], 1)
        return R_

    def pure_svd_predict(self, X):
        """
        prediction for test Matrix X
        :param X: Input Matrix
        :return: prediction vector
        """
        R = self.PSVD()
        pred = []
        for u, i in X:
            pred.append(R[u - 1][i - 1])
        return np.array(pred).flatten()

    @staticmethod
    def rsvd_gradient(Uu, Vi, bu, bi, mu, x, alpha_u, alpha_v, beta_u, beta_v):
        """
        compute the gradient for singular sample x
        :param Uu:
        :param Vi:
        :param bu:
        :param bi:
        :param mu:
        :param x: input sample
        :param alpha_u:
        :param alpha_v:
        :param beta_u:
        :param beta_v:
        :return: gradient of some parameters
        """
        pred = mu + bu + bi + Uu.dot(Vi.T)
        e = pred - x[2]
        grad_mu = e
        grad_bu = e + beta_u * bu
        grad_bi = e + beta_v * bi
        grad_Uu = e * Vi + alpha_u * Uu
        grad_Vi = e * Uu + alpha_v * Vi
        return grad_Uu, grad_Vi, grad_bu, grad_bi, grad_mu

    @staticmethod
    def rsvd(T, X, mu, bu, bi, alpha_u, alpha_v, beta_u, beta_v, gamma):
        """
        regularized SVD with SGD
        :param T: iterations
        :param X: Input Matrix
        :param mu: global rating average
        :param bu: bias of each user
        :param bi: bias of each item
        :param alpha_u:
        :param alpha_v:
        :param beta_u:
        :param beta_v:
        :param gamma:
        :return: some parameters will be used for prediction
        """
        d = 20
        n = np.max(X[:, 0])
        m = np.max(X[:, 1])
        U = (np.random.rand(n, d) - 0.5) * 0.01
        V = (np.random.rand(m, d) - 0.5) * 0.01
        p = X.shape[0]
        perm = np.random.permutation(p)
        X = X[perm]
        for t in range(T):
            for j in range(p):
                x = X[j]
                u, i, r = x
                # compute the gradient
                Uu_g, Vi_g, bu_g, bi_g, mu_g = MatrixFactor.rsvd_gradient(U[u - 1],
                                                                          V[i - 1],
                                                                          bu[u - 1],
                                                                          bi[i - 1],
                                                                          mu, x,
                                                                          alpha_u,
                                                                          alpha_v,
                                                                          beta_u,
                                                                          beta_v)
                # update
                U[u - 1] -= gamma * Uu_g
                V[i - 1] -= gamma * Vi_g
                bu[u - 1] -= gamma * bu_g
                bi[i - 1] -= gamma * bi_g
                mu -= gamma * mu_g
            gamma = 0.9 * gamma
        return U, V, bu, bi, mu

    @staticmethod
    def rsvd_predict(X, U, V, bu, bi, mu):
        """
        use this parameters to predict for input matrix
        :param X: Input Matrix
        :param U:
        :param V:
        :param bu:
        :param bi:
        :param mu:
        :return: prediction vector
        """
        pred = []
        for u, i in X:
            Uu = U[u - 1]
            Vi = V[i - 1]
            buu = bu[u - 1]
            bii = bi[i - 1]
            pred.append(mu + buu + bii + Uu.dot(Vi.T))
        return np.array(pred).flatten()

    @staticmethod
    def pmf_gradient(x, Uu, Vi, alpha_u, alpha_v):
        """
        compute the gradient for singular sample x
        :param x:
        :param Uu:
        :param Vi:
        :param alpha_u:
        :param alpha_v:
        :return:
        """
        r = x[2]
        grad_u = -(r - Uu.dot(Vi.T)) * Vi + alpha_u * Uu
        grad_v = -(r - Uu.dot(Vi.T)) * Uu + alpha_v * Vi
        return grad_u, grad_v

    @staticmethod
    def sgd_pmf(T, X, alpha_u, alpha_v, gamma):
        """
        Probabilistic Matrix Factorization with SGD
        :param T:
        :param X:
        :param alpha_u:
        :param alpha_v:
        :param gamma:
        :return: U,V
        """
        # initial parameters
        d = 20
        n = np.max(X[:, 0])
        m = np.max(X[:, 1])
        U = (np.random.rand(n, d) - 0.5) * 0.01
        V = (np.random.rand(m, d) - 0.5) * 0.01
        p = X.shape[0]
        perm = np.random.permutation(p)
        X = X[perm]

        # SGD
        for t in range(T):
            for i in range(p):
                x = X[i]
                user, item, rating = x
                grad_u, grad_v = MatrixFactor.pmf_gradient(x, U[user - 1], V[item - 1], alpha_u, alpha_v)
                U[user - 1] -= gamma * grad_u
                V[item - 1] -= gamma * grad_v
            gamma = 0.9 * gamma
        return U, V

    @staticmethod
    def pmf_predict(X, U, V):
        """
        predict for input matrix with the parameters learned before
        :param X:
        :param U:
        :param V:
        :return:
        """
        pred = []
        for u, i in X:
            pred.append(U[u - 1].dot(V[i - 1].T))
        return np.array(pred).flatten()


def main():
    # Pure SVD
    data = np.genfromtxt('C:\\Users\\Jack\\Desktop\\Data\\ml-100k\\ml-100k\\u1.base', delimiter='	', dtype=np.int)
    users = data[:, 0]
    items = data[:, 1]
    ratings = data[:, 2]
    svd = MatrixFactor(users, items, ratings)
    test = np.genfromtxt('C:\\Users\\Jack\\Desktop\\Data\\ml-100k\\ml-100k\\u1.test', delimiter='	', dtype=np.int)
    X = test[:, :2]
    y = test[:, 2]
    pred = svd.pure_svd_predict(X)
    print('Pure SVD:')
    rmse1 = MatrixFactor.RMSE(pred, y)
    mae1 = MatrixFactor.MAE(pred, y)
    print('RMSE : {:f}'.format(rmse1))
    print('MAE  : {:f}'.format(mae1))

    # RSVD
    mu = svd.global_average()
    ru = svd.user_average()
    ri = svd.item_average()
    bu, bi = svd.bias(ru, ri)
    alpha_u = alpha_v = beta_u = beta_v = gamma = 0.01
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

    # PMF
    iterations = 10
    alpha_u = alpha_v = gamma = 0.01
    U, V = MatrixFactor.sgd_pmf(iterations, data[:, :3], alpha_u, alpha_v, gamma)
    pred = MatrixFactor.pmf_predict(X, U, V)
    rmse3 = MatrixFactor.RMSE(pred, y)
    mae3 = MatrixFactor.MAE(pred, y)
    print('PMF')
    print('RMSE :{:f}'.format(rmse3))
    print('MAE  :{:f}'.format(mae3))

    # plot
    plot_data = np.array([rmse1, mae1, rmse2, mae2, rmse3, mae3]).reshape(3, 2)
    labels = ('PSVD', 'RSVD', 'PMF')
    MatrixFactor.plot(plot_data, labels)


if __name__ == '__main__':
    main()
