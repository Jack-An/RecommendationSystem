import numpy as np
import operator
from ml_in_action.visiual_tree import DrawTree


def calc_shannon_ent(dataset):
    m, n = dataset.shape
    label = {}
    for sample in dataset:
        current_label = sample[-1]  # 计算每个分类的样本数（label）
        if current_label not in label.keys():
            label[current_label] = 0
        label[current_label] += 1
    shano_ent = 0.0
    for key in label:
        prob = label[key] / m
        shano_ent -= prob * np.log2(prob)
    return shano_ent


def create_dataset():
    dataSet = np.array([
        # 1
        ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
        # 2
        ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '好瓜'],
        # 3
        ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
        # 4
        ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '好瓜'],
        # 5
        ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
        # 6
        ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '好瓜'],
        # 7
        ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', '好瓜'],
        # 8
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', '好瓜'],
        # 9
        ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', '坏瓜'],
        # 10
        ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', '坏瓜'],
        # 11
        ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', '坏瓜'],
        # 12
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', '坏瓜'],
        # 13
        ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', '坏瓜'],
        # 14
        ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', '坏瓜'],
        # 15
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '坏瓜'],
        # 16
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', '坏瓜'],
        # 17
        ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', '坏瓜']
    ])
    labels = np.array(['色泽', '根蒂', '敲声', '纹理', '脐部', '触感'])
    return dataSet, labels


def create():
    data = np.array([[1, 1, 'yes'],
                     [1, 1, 'yes'],
                     [1, 0, 'no'],
                     [0, 1, 'no'],
                     [0, 1, 'no']])
    labels = np.array(['no surfing', 'flippers'])
    return data, labels


def load_data():
    with open('../ml_in_action/lenses.txt') as f:
        lenses = [inst.strip().split('\t') for inst in f.readlines()]
    labels = np.array(['age', 'prescript', 'astigmatic', 'tearRate'])
    return np.array(lenses), labels


def split_dataset(dataset, axis, value):
    sub_dataset = []
    for feature in dataset:
        if feature[axis] == value:
            reduced = feature
            reduced = np.delete(reduced, axis)
            sub_dataset.append(reduced)
    return np.array(sub_dataset)


def choose_best_feature(dataset):
    num_features = dataset.shape[1] - 1  # 除去最后一列的标签
    base_entropy = calc_shannon_ent(dataset)
    best_info_gain = 0.0
    best_feature = -1
    for i in range(num_features):
        feature_list = dataset[:, i]
        unique_vals = np.unique(feature_list)
        new_entropy = 0.0
        for value in unique_vals:
            sub_dataset = split_dataset(dataset, i, value)
            prob = sub_dataset.shape[0] / dataset.shape[0]
            new_entropy += prob * calc_shannon_ent(sub_dataset)
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature


def majority_cnt(class_list):
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def create_tree(dataset, labels):
    class_list = dataset[:, -1]
    if np.where(class_list == class_list[0])[0].size == class_list.size:  # 如果所有的类标签完全相同，则返回这个类标签
        return class_list[0]
    if dataset[0].size == 1:  # 如果已经是最后一个特征，并且还存在多个类标签，则采取多数表决的方式生成叶节点
        return majority_cnt(class_list)
    best_feature = choose_best_feature(dataset)
    best_feature_label = labels[best_feature]
    tree = {best_feature_label: {}}
    labels = np.delete(labels, best_feature)
    features = dataset[:, best_feature]  # 得到最好的划分属性的所有取值
    unique_vals = np.unique(features)
    for value in unique_vals:  # 递归构造树
        sub_labels = labels[:]
        tree[best_feature_label][value] = create_tree(split_dataset(dataset, best_feature, value), sub_labels)
    return tree


def classify(input_data, tree, labels):
    if not isinstance(tree, dict):
        return tree
    label = list(tree.keys())[0]
    label_index = list(labels).index(label)
    # numpy将数据集全都转成了字符串，所以这里需要转成字符
    return classify(input_data, tree[label][str(input_data[label_index])], labels)


def store_tree(tree, filename):
    import pickle
    with open(filename, 'wb') as f:
        pickle.dump(tree, f)


def grab_tree(filename):
    import pickle
    with open(filename, 'rb') as f:
        return pickle.load(f)


def main():
    # data, labels = create_dataset()
    # tree = create_tree(data, labels)
    # draw = DrawTree(tree)
    # draw.create_plot()
    data, labels = load_data()
    tree = create_tree(data, labels)
    draw = DrawTree(tree)
    draw.create_plot()
    filename = '../ml_in_action/test_tree.txt'
    store_tree(tree, filename)
    print(grab_tree(filename))


if __name__ == '__main__':
    main()
