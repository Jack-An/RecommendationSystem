import numpy as np
import scipy.optimize as op


def load_data(filename):
    data = np.genfromtxt(filename, delimiter=',', dtype=np.int)
    data = data[:, :3]
    return data


def data2matrix(data, num_user, num_item):
    R = np.zeros((num_item, num_user), dtype=np.int)
    Y = np.zeros(R.shape, dtype=np.float32)
    for i, j, r in data:
        R[j - 1][i - 1] = 1
        Y[j - 1][i - 1] = r
    return R, Y


def normalization(Y, R):
    """
    normalize the rating matrix Y
    make the rating for each item is zero
    :param Y:rating matrix
    :param R:if real rating
    :return:Y_norm,Y_mean
    """
    m, n = Y.shape
    Y_norm = np.zeros(Y.shape)
    Y_mean = np.zeros(m)
    for i in range(m):
        idx = np.where(R[i, :] == 1)[0]
        if idx.size != 0:
            Y_mean[i] = np.mean(Y[i, idx])
            Y_norm[i, idx] = Y[i, idx] - Y_mean[i]
    return Y_norm, Y_mean


def cost_func(params, Y, R, num_user, num_item, num_feature, lambda_):
    """
    calculate the gradient and cost
    :param params: input params
    :param Y: rating matrix
    :param R: if real rating
    :param num_user: users
    :param num_item: movies
    :param num_feature: features
    :param lambda_: regularization parameters
    :return: cost,gradient
    """
    X = np.reshape(params[:num_item * num_feature], (num_item, num_feature))
    Theta = np.reshape(params[num_item * num_feature:], (num_user, num_feature))
    errors = (X.dot(Theta.T) - Y) * R
    regularization_X = lambda_ * np.sum(np.sum(np.square(X))) / 2
    regularization_Theta = lambda_ * np.sum(np.sum(np.square(Theta))) / 2
    J = np.sum(np.sum(np.square(errors))) / 2 + regularization_Theta + regularization_X
    X_grad = errors.dot(Theta) + lambda_ * X
    Theta_grad = errors.T.dot(X) + lambda_ * Theta
    return J, np.hstack((X_grad.flatten(), Theta_grad.flatten()))


def prediction(X, Theta, X_test, Y_mean):
    pred = []
    for u, i in X_test:
        pred.append(np.dot(X[i - 1, :], Theta[u - 1, :].T) + Y_mean[i - 1])
    return np.array(pred).flatten()


def RMSE(pred, y_test):
    return np.sqrt(np.mean(np.square(y_test - pred)))


def MAE(pred, y_test):
    return np.mean(np.abs(y_test - pred))


def main():
    # load data
    filename = 'C:/Users/Jack/Desktop/Data/ml/u1.base.csv'
    data = load_data(filename)
    # setup parameters
    num_user = 943
    num_item = 1682
    num_feature = 10
    R, Y = data2matrix(data, num_user, num_item)
    Y_norm, Y_mean = normalization(Y, R)
    X = np.random.rand(num_item, num_feature)
    Theta = np.random.rand(num_user, num_feature)
    params = np.hstack((X.flatten(), Theta.flatten()))
    lambda_ = 10
    cost = lambda t: cost_func(t, Y_norm, R, num_user, num_item, num_feature, lambda_)
    # optimize
    result = op.minimize(fun=cost, args=(), x0=params, jac=True, method='TNC', options={'maxiter': 200})
    theta = result.x
    # recover parameters
    X = np.reshape(theta[:num_item * num_feature], (num_item, num_feature))
    Theta = np.reshape(theta[num_item * num_feature:], (num_user, num_feature))
    # load test data
    filename = 'C:/Users/Jack/Desktop/Data/ml/u1.test.csv'
    test_data = load_data(filename)
    X_test = test_data[:, :2]
    y_test = test_data[:, 2]
    # prediction and accuracy
    pred = prediction(X, Theta, X_test, Y_mean)
    print('Collaborative CF (coursera) ')
    print('RMSE :{:f}'.format(RMSE(pred, y_test)))
    print('MAE : {:f}'.format(MAE(pred, y_test)))


if __name__ == '__main__':
    main()
