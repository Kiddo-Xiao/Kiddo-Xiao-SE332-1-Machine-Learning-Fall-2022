import time
import pandas
import numpy as np
import statsmodels.api as sm  # for finding the p-value
from sklearn import preprocessing
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

# 常量
max_iterations = 5000 # 最大迭代次数
reg_strength = 10000 # 正则化强度
learning_rate = 0.000001 # 学习率
cost_threshold = 0.0001  # 阈值设置（百分比方式：将Δcost与当前cost的1%比较）

def load_data():
    # print("load data...")
    path = ['./data/X_train.csv',  './data/X_test.csv', './data/Y_train.csv', './data/Y_test.csv']
    X_train = pandas.read_csv(path[0]).values
    X_test = pandas.read_csv(path[1]).values
    Y_train = pandas.read_csv(path[2])['Label'].values
    Y_test = pandas.read_csv(path[3])['Label'].values
    # Y输入为01，SVM中为+-1!!!
    Y_train = Y_train * 2 - 1
    Y_test = Y_test * 2 - 1
    # 预处理feature
    X_train_normal = preprocessing.MinMaxScaler().fit_transform(X_train)
    X_train = pandas.DataFrame(X_train_normal)
    X_test_normal = preprocessing.MinMaxScaler().fit_transform(X_test)
    X_test = pandas.DataFrame(X_test_normal)
    # 给X_train添加一列值为1，为了将截距b推入权重向量w中
    X_train.insert(loc=len(X_train.columns), column='intercept', value=1)
    X_train=X_train.to_numpy()
    # X_test格式需与X_train统一！！！
    X_test.insert(loc=len(X_test.columns), column='intercept', value=1)
    X_test=X_test.to_numpy()
    return X_train,X_test,Y_train,Y_test


def compute_cost(W, X, Y):
    # print("compute cost...")
    # 计算损失函数 hinge loss
    N = X.shape[0]
    distances = 1 - Y * (np.dot(X, W))
    distances[distances < 0] = 0  # = max(0, distance)
    hinge_loss = reg_strength * (np.sum(distances) / N)

    # 计算 cost
    cost = 1 / 2 * np.dot(W, W) + hinge_loss
    return cost

def calculate_cost_gradient(W, X, Y):
    # print(W)
    # if only one example is passed (eg. in case of SGD)
    if type(Y) == np.int64:
        Y = np.array([Y])
        X = np.array([X])
    distance = 1 - (Y * np.dot(X, W))
    # 计算梯度的改变量dw
    dw = np.zeros(len(W))

    for index, d in enumerate(distance):
        if max(0, d) == 0:
            di = W
        else:
            di = W - (reg_strength * Y[index] * X[index])
            # print(di)
        dw += di
    # print(dw)
    dw = dw/len(Y)  # average
    return dw

#输入测试集XY，得到最佳权重W
def sgd(features_x, outputs_y):
    weights = np.zeros(features_x.shape[1])
    nth = 0
    prev_cost = float("inf")
    # 随机梯度下降
    for iter_time in range(1, max_iterations):
        # shuffle 防止重复更新
        X, Y = shuffle(features_x, outputs_y)
        for index, x in enumerate(X):
            # 得到梯度改变量
            ascent = calculate_cost_gradient(weights, x, Y[index])
            # print(ascent)
            # 梯度更新：w = w - η * dw  （η：步长，即学习率)
            weights = weights - (learning_rate * ascent)

        # 在第 2^nth 次迭代进行收敛检验：迭代之间的改进不大于一个小的阈值
        if iter_time == 2 ** nth or iter_time == max_iterations - 1:
            cost = compute_cost(weights, features_x, outputs_y)
            print("Iter_time is:{} and Cost is: {}".format(iter_time, cost))
            # 终止迭代：cost变化量 < 当前cost*0.01
            if abs(prev_cost - cost) < cost_threshold * prev_cost:
                return weights
            prev_cost = cost
            nth += 1

    return weights

# test = 0 ? 测X_test ; 测X_train
def svm_gradient(test):
    # 读数据
    X_train,X_test,Y_train,Y_test = load_data()
    # 开始计时(leaddata不计入！)
    time_sta = time.time()

    # 训练
    print("training started...")
    W = sgd(X_train, Y_train)
    print("training finished...")
    # print("best weights are: {}".format(W))
    # 预测
    Y_pred = np.array([])
    if test == 0:
        for i in range(X_test.shape[0]):
            yp = np.sign(np.dot(W, X_test[i]))  # model
            Y_pred = np.append(Y_pred, yp)
        # 输出结果
        test_size = len(X_test)
        mispred_num = (Y_test != Y_pred).sum()
    if test == 1:
        for i in range(X_train.shape[0]):
            yp = np.sign(np.dot(W, X_train[i]))  # model
            Y_pred = np.append(Y_pred, yp)
        # 输出结果
        test_size = len(X_train)
        mispred_num = (Y_train != Y_pred).sum()
    correct_rate = 1 - mispred_num / test_size
    print("INFO: Number of mislabeled points out of a total %d points : %d, with correct rate %f"
          % (test_size, mispred_num, correct_rate))
    # 终止计时
    time_end = time.time()
    # 耗时
    time_all = time_end - time_sta
    print("INFO: svm(sklearn) - time cost : %f s" % time_all)
    return correct_rate, time_all



if __name__ == '__main__':
    N = 10
    correct_rate_tot = 0.0
    time_tot = 0.0
    # 多次训练+预测X_test求平均
    for i in range(N):
        print("--------------------NO."+str(i+1)+"--------------------")
        cr, tt = svm_gradient(0)
        correct_rate_tot += cr
        time_tot += tt

    avg_cr_tot = correct_rate_tot / N
    avg_tt = time_tot / N
    print("--------------------[Result of test_data]--------------------")
    print("INFO: Result: %d times, %f average correct rate, %f average time consume" % (N, avg_cr_tot, avg_tt))

    correct_rate_tot = 0.0
    time_tot = 0.0
    # 多次预测X_train求平均
    for i in range(N):
        print("--------------------NO."+str(i+1)+"--------------------")
        cr, tt = svm_gradient(1)
        correct_rate_tot += cr
        time_tot += tt

    avg_cr_tot = correct_rate_tot / N
    avg_tt = time_tot / N
    print("--------------------[Result of train_data]--------------------")
    print("INFO: Result: %d times, %f average correct rate, %f average time consume" % (N, avg_cr_tot, avg_tt))

