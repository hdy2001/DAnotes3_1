import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


def test(Y, y_pred):
    """
    Y: 标签Size([1000])
    y_pred: 预测结果Size([1000])
    """
    pred = y_pred
    answer = (Y == pred)

    # 计算TP，FP，FN，TN
    TP = np.sum(np.where((Y == pred) & pred, np.ones_like(Y),
                         np.zeros_like(Y)))
    FP = np.sum(np.where((Y != pred) & pred, np.ones_like(Y),
                         np.zeros_like(Y)))
    FN = np.sum(
        np.where((Y != pred) & (pred == 0), np.ones_like(Y), np.zeros_like(Y)))
    TN = np.sum(
        np.where((Y == pred) & (pred == 0), np.ones_like(Y), np.zeros_like(Y)))

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
    auROC = metrics.roc_auc_score(Y, y_pred, 'ovo')
    auPRC = metrics.average_precision_score(Y, y_pred, 'ovo')
    print("auROC为：")
    print(auROC)
    print("auPRC为：")
    print(auPRC)


def test_sklearn(Y, y_pred):
    """
    Y: 标签Size([1000])
    y_pred: 预测结果Size([1000])
    """
    pred = y_pred
    answer = (Y == pred)

    # 使用Accruracy方法
    ans = np.sum(answer) / len(answer)
    print("使用Accruracy方法准确率为：")
    print(ans)

    # 计算recall，precision，F1
    recall = recall_score(Y, pred, average='macro')
    precision = precision_score(Y, pred, average="macro")
    F1 = f1_score(Y, pred, average="macro")
    print("recall-marcro为：")
    print(recall)
    print("precision-marcro为：")
    print(precision)
    print("F1-marcro为：")
    print(F1)

    recall = recall_score(Y, pred, average='micro')
    precision = precision_score(Y, pred, average="micro")
    F1 = f1_score(Y, pred, average="micro")
    print("recall-micro为：")
    print(recall)
    print("precision-micro为：")
    print(precision)
    print("F1-micro为：")
    print(F1)