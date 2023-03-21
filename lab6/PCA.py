import numpy as np
import matplotlib.pyplot as plt
import math

# read data


def readfile(filename):
    with open(filename) as f:
        lines = f.read()
        lines = lines.split("\n")
        for i in range(len(lines)):
            lines[i] = lines[i].split("\t")
        date = []
        temp = []
        for i in range(len(lines)):
            temp.append(float(lines[i][-1]))
        for i in range(len(lines)):
            date.append(lines[i][0]+lines[i][1]+lines[i][2])
    return date, temp


def F2C(temp):
    for i in range(len(temp)):
        temp[i] = (temp[i]-32)*5/9
    return temp


def mean_square_error(X, XP):
    error = 0
    n = len(XP)
    for i in range(n):
        error += (X[i]-XP[i])**2
    error = error/n
    return error


def data_trans(data):
    A = np.zeros((len(data), 2))
    for i, j in enumerate(data):
        A[i][0] = i
        A[i][1] = j
    return A


def norm_data(data):
    X = data.copy()
    ma = max(data)
    mi = min(data)
    for i in range(len(X)):
        X[i] = (X[i]-mi)/(ma-mi)
    return X


def inverse_norm_data(data, X):
    A = X.copy()
    ma = max(data)
    mi = min(data)
    for i in range(len(A)):
        A[i] = A[i]*(ma-mi)+mi
    return A


class PCA:
    def __init__(self, tn):
        self.tn = tn

    def mean(self, data):
        n = len(data)
        x = 0
        y = 0
        for i in data:
            x += i[0]
            y += i[1]
        m = np.array([x/n, y/n])
        return m

    def subtract_mean(self, data, mean):
        A = data.copy()
        for i in A:
            i[0] = i[0]-mean[0]
            i[1] = i[1]-mean[1]
        return A

    def find_c(self, V, mean, x1):
        return (x1-mean[0])/V[0]

    def find_yp(self, V, mean, c):
        return mean[1]+c*V[1]

    def prediction(self, data, n):
        XP = np.array([])
        for i in range(n):
            A = data[i-365-self.tn:i-365]
            m = self.mean(A)
            A = self.subtract_mean(A, m)
            U, S, V = np.linalg.svd(A)
            c = self.find_c(V[0], m, i+365)
            y = self.find_yp(V[0], m, c)
            XP = np.append(XP, y)
        return XP


filename = "temp2.txt"
date, temp = readfile(filename)
temp = np.array(temp)
temp = F2C(temp)

n = 365
tn = 42
# 42

# data
train_temp = temp[:-2*n]
test_temp = temp[-2*n:]
test_temp_2018 = temp[-n:]

# without normalize
t = data_trans(test_temp)

model = PCA(tn)
XP = model.prediction(t, 365)
print(f"XP mse without normalize: {mean_square_error(test_temp_2018, XP)}")

plt.plot(test_temp_2018, label="2018")
plt.plot(XP, label="XP")
plt.legend()
plt.figure()

# with normalize
X = norm_data(test_temp)
t = data_trans(X)

model = PCA(tn)
XP = model.prediction(t, 365)
XP = inverse_norm_data(test_temp, XP)
print(f"XP mse with normalize: {mean_square_error(test_temp_2018, XP)}")


plt.plot(test_temp_2018, label="2018")
plt.plot(XP, label="XP norm")
plt.legend()

plt.show()

# X = norm_data(test_temp)
# t = data_trans(X)

# m = 100
# midx = -1
# for tn in range(10, 365):
#     model = PCA(tn)
#     XP = model.prediction(t, 365)
#     XP = inverse_norm_data(test_temp, XP)
#     mse = mean_square_error(test_temp_2018, XP)
#     print(f"{tn} {mse}")
#     if mse < m:
#         m = mse
#         midx = tn
# print(f"minidx: {midx} min mse: {m}")
