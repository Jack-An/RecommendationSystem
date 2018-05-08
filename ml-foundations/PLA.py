import numpy as np


def pla(data):
    theta = np.zeros(data.shape[1])
    data = np.c_[np.ones(data.shape[0]), data]
    iterations = 0
    while True:
        pred = np.sign(data[:, :-1].dot(theta.reshape(theta.size, 1)))
        equal = pred.flatten() - data[:, -1]
        si = np.where(equal != 0)[0]  # 找到误分类的数据
        if si.size != 0:
            iterations += 1
            idx = np.random.choice(si, 1)[0]  # 随机选取一个误分类的数据
            theta += data[idx][:-1] * data[idx][-1]
        else:  # 直到没有误分类的数据，然后退出循环
            break
    return theta, iterations


def pocket_pla(data):
    theta = np.zeros(data.shape[1])
    data = np.c_[np.ones(data.shape[0]), data]
    current_theta = theta
    error = cost_function(theta, data)
    iterations = 1500
    for i in range(iterations):
        pred = np.sign(data[:, :-1].dot(theta.reshape(theta.size, 1)))
        equal = pred.flatten() - data[:, -1]
        si = np.where(equal != 0)[0]  # 找到误分类的数据
        if si.size != 0:
            iterations += 1
            idx = np.random.choice(si, 1)[0]  # 随机选取一个误分类的数据
            current_theta += data[idx][:-1] * data[idx][-1]
            current_error = cost_function(current_theta, data)
            if current_error < error:
                error = current_error
                theta = current_theta

    return theta, error


def cost_function(theta, data):
    pred = np.sign(data[:, :-1].dot(theta.reshape(theta.size, 1)))
    equal = pred.flatten() - data[:, -1]
    si = np.where(equal != 0)[0]  # 找到误分类的数据
    return si.size / data.shape[0]


def main():
    data = np.genfromtxt('../data/pocket_train.dat')
    theta, error = pocket_pla(data)
    print(error)
    print(theta)
    test = np.genfromtxt('../data/pocket_test.dat')
    test = np.c_[np.ones(test.shape[0]), test]
    print(cost_function(theta, test))


if __name__ == '__main__':
    main()
