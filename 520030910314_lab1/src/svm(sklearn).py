import time
import pandas
import sklearn
from sklearn.svm import SVC
def load_data():
    path = ['./data/X_train.csv',  './data/X_test.csv', './data/Y_train.csv', './data/Y_test.csv']
    X_train = pandas.read_csv(path[0]).values
    X_test = pandas.read_csv(path[1]).values
    Y_train = pandas.read_csv(path[2])['Label'].values
    Y_test = pandas.read_csv(path[3])['Label'].values
    return X_train,X_test,Y_train,Y_test

def svm_sklearn(kernel):
    # 读数据
    X_train,X_test,Y_train,Y_test = load_data()
    # 开始计时(leaddata不计入！)
    time_sta = time.time()

    # 训练
    svm = SVC(kernel=kernel)
    svm.fit(X_train, Y_train)
    # 预测
    Y_pred = svm.predict(X_test)
    # 输出结果
    test_size = len(X_test)
    mispred_num = (Y_test != Y_pred).sum()
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

    # 选择不同kernal函数
    for fig_num, kernel in enumerate(('linear', 'poly', 'rbf')):
        correct_rate_tot = 0.0
        time_tot = 0.0
        # 多次训练+预测求平均
        for i in range(N):
            cr, tt = svm_sklearn(kernel)
            correct_rate_tot += cr
            time_tot += tt

        avg_cr_tot = correct_rate_tot / N
        avg_tt = time_tot / N
        print("--------------------kernal func : "+kernel+"--------------------")
        print("INFO: Result: %d times, %f average correct rate, %f average time consume" % (N, avg_cr_tot, avg_tt))
        print("------------------------------------------------------------")
