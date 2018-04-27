import numpy as np


def data_precess(filenama):
    data = np.genfromtxt(filenama, delimiter=',', dtype=np.int)
    index = np.where(data[:, 2] >= 4)
    return data[index][:, :3]


def precision(Ire, Ite, test_users, K):
    prec = []
    for u in test_users:
        re = Ire[u - 1]
        te = Ite[u - 1]
        num = np.intersect1d(re, te).size
        prec.append(num / K)
    return prec, np.mean(prec)


def recall(Ire, Ite, test_users, K):
    rec = []
    for u in test_users:
        re = Ire[u - 1]
        te = Ite[u - 1]
        num = np.intersect1d(re, te).size
        rec.append(num / len(Ite[u - 1]))
    return rec, np.mean(rec)


def F1(pre, rec):
    f1 = []
    for p, r in zip(pre, rec):
        sign = p + r
        if sign != 0:
            f1.append(2 * p * r / sign)
        else:
            f1.append(sign)
    return f1, np.mean(f1)


def NDCG(Ire, Ite, test_users, K):
    ndcg = []
    zu = 0
    for i in range(K):
        zu += 1 / np.log2(i + 2)
    for u in test_users:
        re = Ire[u - 1]
        te = Ite[u - 1]
        dcg = 0
        for i in range(K):
            if re[i] in te:
                dcg += (1 / np.log2(i + 2))
        ndcg.append(dcg / zu)
    return ndcg, np.mean(ndcg)


def one_call(Ire, Ite, test_users, K):
    one = []
    for u in test_users:
        re = Ire[u - 1]
        te = Ite[u - 1]
        num = np.intersect1d(re, te).size
        if num >= 1:
            one.append(1)
        else:
            one.append(0)
    return one, np.mean(one)


def MRR(Ire, Ite, test_users):
    mrr = []
    for u in test_users:
        re = Ire[u - 1]
        te = Ite[u - 1]
        p = 0
        for i in range(len(re)):
            if re[i] in te:
                p = i + 1
                break
        if p != 0:
            mrr.append(1 / p)
        else:
            mrr.append(0)
    return mrr, np.mean(mrr)


def MAP(Ire, Ite, test_usrs):
    mp = []
    for u in test_usrs:
        re = Ire[u - 1]
        te = Ite[u - 1]
        ap = 0
        for i in te:
            p = np.where(re == i)[0] + 1
            tmp = 0
            for j in te:
                if np.where(re == j)[0] < np.where(re == i)[0]:
                    tmp += 1
            val = (tmp + 1) / p
            ap += val
        mp.append(ap / len(te))
    return mp, np.mean(mp)


def ARP(Ire, Ite, test_users):
    arp = []
    for u in test_users:
        re = Ire[u - 1]
        te = Ite[u - 1]
        rp = 0
        for i in te:
            p = np.where(re == i)[0] + 1
            rp += (p / len(re))
        arp.append(rp / len(te))
    return arp, np.mean(arp)


def AUC(bi, Rte, test_users):
    auc = []
    for u in test_users:
        te = Rte[u - 1]
        num = 0
        for i, j in te:
            if bi[i] > bi[j]:
                num += 1
        auc.append(num / len(te))
    return auc, np.mean(auc)


def __Rte__(train_data, test_data, test_users):
    Rte = {}
    I = np.arange(1682, dtype=np.int)
    for u in test_users:
        data1 = train_data[np.where(train_data[:, 0] == u)][:, 1] - 1
        data2 = test_data[np.where(test_data[:, 0] == u)][:, 1] - 1
        pair = []
        R = np.union1d(data1, data2)
        data3 = np.setdiff1d(I, R)
        for i in data2:
            for j in data3:
                pair.append((i, j))
        Rte[u - 1] = pair
    return Rte


# bias of item
def bias(data, num_users, num_items):
    bi = []
    mu = data.shape[0] / (num_users * num_items)
    for i in range(num_items):
        id = np.where(data[:, 1] == i + 1)[0]
        bi.append(id.size / num_users - mu)
    return bi


# 已经购买过的集合Iu
def item_set(data, num_users):
    Iu = []
    for i in range(num_users):
        idx = np.where(data[:, 0] == i + 1)
        Iu.append(data[:, 1][idx] - 1)
    return Iu


# 推荐的商品
def recommend(Iu, bi, num_users, K=None):
    Ire = {}
    I = np.argsort(bi)[::-1]
    for i in range(num_users):
        re = I
        for u in Iu[i]:
            id = np.where(re == u)
            re = np.delete(re, id)
        Ire[i] = re[:K]
    return Ire


# 计算测试集的Iu^te
def real_set(test, test_users):
    Ite = {}
    for u in test_users:
        idx = np.where(test[:, 0] == u)
        Ite[u - 1] = test[:, 1][idx] - 1
    return Ite


def main():
    trainfile = 'C:/Users/Jack/Desktop/Data/ml/u1.base.csv'
    testfile = 'C:/Users/Jack/Desktop/Data/ml/u1.test.csv'
    num_users = 943
    num_items = 1682
    data = data_precess(trainfile)  # load train data
    Iu = item_set(data, num_users)
    # compute parameters
    bi = bias(data, num_users, num_items)
    test = data_precess(testfile)  # load test data
    Ire = recommend(Iu, bi, num_users, 5)
    test_users = np.unique(test[:, 0])
    Ite = real_set(test, test_users)
    # Ranking Evaluation
    prec = precision(Ire, Ite, test_users, 5)
    print('Pre@5 : {:f}'.format(prec[1]))
    rec = recall(Ire, Ite, test_users, 5)
    print('Rec@5 : {:f}'.format(rec[1]))
    f1 = F1(prec[0], rec[0])
    print('F1@5  : {:f}'.format(f1[1]))
    ndcg = NDCG(Ire, Ite, test_users, 5)
    print('NDCG@5: {:f}'.format(ndcg[1]))
    one = one_call(Ire, Ite, test_users, 5)
    print('1-call@5: {:f}'.format(one[1]))
    mrr = MRR(recommend(Iu, bi, num_users), Ite, test_users)
    print('MRR   : {:f}'.format(mrr[1]))
    mp = MAP(recommend(Iu, bi, num_users), Ite, test_users)
    print('MAP   : {:f}'.format(mp[1]))
    arp = ARP(recommend(Iu, bi, num_users), Ite, test_users)
    print('ARP   : {:f}'.format(arp[1]))
    Rte = __Rte__(data, test, test_users)
    auc = AUC(bi, Rte, test_users)
    print('AUC   : {:f}'.format(auc[1]))


if __name__ == '__main__':
    main()
