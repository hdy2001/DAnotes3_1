import scipy.io as scio
import random
import numpy as np
import os
from sklearn import metrics, linear_model


def shuffle(X, Y, seed=0):
    """
    随机打乱原始数据
    """
    random.seed(0)
    index = [i for i in range(X.shape[0])]
    random.shuffle(index)
    return X[index], Y[index]


def get_zero_and_one(X, Y):
    """
    从给定数据中抽取数字0和数字1的样本
    """
    index_1 = Y == 0
    index_8 = Y == 1
    index = index_1 + index_8
    return X[index], Y[index]


def load_data(data_dir="./", data_file="mnist.mat"):
    # 加载数据，划分数据集
    data = scio.loadmat(os.path.join(data_dir, data_file))
    train_X, test_X = data['train_X'], data['test_X']
    train_Y, test_Y = data['train_Y'].reshape(
        train_X.shape[0]), data['test_Y'].reshape(test_X.shape[0])

    # 从训练数据中抽取数字 0 和数字 1 的样本，并打乱
    train_X, train_Y = get_zero_and_one(train_X, train_Y)
    train_X, train_Y = shuffle(train_X, train_Y)
    train_Y = (train_Y == 1).astype(np.float32)  # 1->True 0->false
    # 从测试数据中抽取数字 0 和数字 1 的样本，并打乱
    test_X, test_Y = get_zero_and_one(test_X, test_Y)
    test_X, test_Y = shuffle(test_X, test_Y)
    test_Y = (test_Y == 1).astype(np.float32)
    print("原始图片共有%d张，其中数字1的图片有%d张。" % (test_X.shape[0], sum(test_Y == 1)))
    return train_X, train_Y, test_X, test_Y


def ext_feature(train_X, test_X):
    """
    抽取图像的白色像素点比例作为特征
    """
    train_feature = np.sum(train_X > 200, axis=1) / 784
    test_feature = np.sum(test_X > 200, axis=1) / 784
    return train_feature, test_feature


def train(w, b, X, Y, alpha=0.1, epochs=200, batchsize=32):
    """
    YOUR CODE HERE
    """
    def pi_x(XX, ww, bb):
        return 1 / (np.exp(-(ww * XX + bb)) + 1)

    for i in range(epochs):
        shuffle(X, Y)
        Batch = X.shape[0] // batchsize
        for j in range(Batch):
            train_x = X[i * batchsize:(i + 1) * batchsize]
            train_y = Y[i * batchsize:(i + 1) * batchsize]
            db = alpha * np.sum(train_y - pi_x(train_x, w, b)) / batchsize
            dw = alpha * np.sum(
                (train_y - pi_x(train_x, w, b)) * train_x) / batchsize
            w += dw
            b += db
    return w, b


def test(Y, y_pred):
    """
    YOUR CODE HERE
    """
    pred = (y_pred > 0.5)
    answer = (Y == pred)

    # 计算TP，FP，FN，TN
    TP = np.sum(np.where((Y == pred) & pred, np.ones_like(Y), np.zeros_like(Y)))
    FP = np.sum(np.where((Y != pred) & pred, np.ones_like(Y), np.zeros_like(Y)))
    FN = np.sum(np.where((Y != pred) & (pred == 0), np.ones_like(Y), np.zeros_like(Y)))
    TN = np.sum(np.where((Y == pred) & (pred == 0), np.ones_like(Y), np.zeros_like(Y)))

    # 使用Accruracy方法
    ans = np.sum(answer) / len(answer)
    print("使用Accruracy方法准确率为：")
    print(ans)
    # 使用BER方法
    BER = 1 / 2 * (FP / (FP + TN) + FN / (FN + TP))
    print("BER为：")
    print(BER)

    # 使用MCC方法
    A = TP * TN - FP * FN
    # 防止溢出
    B = np.sqrt((TP + FP) * (FP + TN)) * np.sqrt((TN + FN) * (FN + TP))
    MCC = A / (B + 1e-8)
    print("MCC为：")
    print(MCC)

    # 计算sensitivity和specificity
    sensitivity = TP / (TP + FN)
    specificity = TN / (FP + TN)
    print("sensitivity为：")
    print(sensitivity)
    print("specificity为：")
    print(specificity)

    # 计算recall，precision，F1
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    F1 = 2 / (1 / precision + 1 / recall)
    print("recall为：")
    print(recall)
    print("precision为：")
    print(precision)
    print("F1为：")
    print(F1)

    # 计算auROC，auPRC
    auROC = metrics.roc_auc_score(Y, y_pred)
    auPRC = metrics.average_precision_score(Y, y_pred)
    print("auROC为：")
    print(auROC)
    print("auPRC为：")
    print(auPRC)

def pi_x(w, b, XX):
    return 1 / (np.exp(-(w * XX + b)) + 1)


if __name__ == "__main__":
    # 加载数据
    train_X, train_Y, test_X, test_Y = load_data()
    # 抽取特征
    train_feature, test_feature = ext_feature(train_X, test_X)

    # 随机初始化参数
    w = np.random.randn()
    b = np.random.randn()

    # 训练及测试
    """
    YOUR CODE HERE
    """
    w, b = train(w, b, train_feature, train_Y)
    test(test_Y, pi_x(w, b, test_feature))

    models = linear_model.LogisticRegression(random_state=0).fit(train_feature.reshape(-1, 1), train_Y.reshape(-1, 1))
    print("采用线性模型：")
    test(test_Y, models.predict(test_feature.reshape(-1, 1)).reshape(-1))