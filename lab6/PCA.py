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


def normal_distribution(x, mean, sigma):
    return np.exp(-1*((x-mean)**2)/(2*(sigma**2)))/(math.sqrt(2*np.pi) * sigma)


def mean_square_error(X, XP):
    error = 0
    n = len(XP)
    for i in range(n):
        error += (X[i]-XP[i])**2
    error = error/n
    return error


def norm_weight_array(mean, sigma, n):
    x = np.arange(0, n)
    w = normal_distribution(x, mean, sigma)
    return w


def linear_weight_array(slope, bias, n):
    x = np.arange(0, n)
    w = x*(np.ones(n)*slope)+np.ones(n)*bias
    return w


def weight_array(low, high, t, n):
    w = np.ones(n)*low
    for i in range(t):
        w[-i-1] = high
    return w


class PredictionModel:
    def __init__(self, M):
        self.M = M

    # model
    def find_optimal_value(self, X, w):
        nL = len(X)
        A = np.zeros((self.M+1, self.M+1))
        B = np.zeros(self.M+1)

        # A
        for i in range(self.M+1):
            for j in range(self.M+1):
                x = 0
                for k in range(nL):
                    x += (k**(i+j))*w[k]
                A[i][j] = x

        # B
        for i in range(self.M+1):
            x = 0
            for k in range(nL):
                x += ((k**i)*X[k])*w[k]
            B[i] = x

        optimalVal = np.linalg.inv(A).dot(B)
        return optimalVal

    def curvefitting(self, ak, n):
        xp = 0
        for i in range(self.M+1):
            xp += ak[i]*(n**i)
        return xp

    def prediction(self, data, trainN, w, N):
        XP = np.array([])

        for i in range(N):
            trainingData = data[i:i+trainN]
            ak = self.find_optimal_value(trainingData, w)
            XP = np.append(XP, self.curvefitting(ak, trainN))

        return XP


filename = "temp2.txt"
date, temp = readfile(filename)
temp = np.array(temp)
temp = F2C(temp)

n = 365

# data
train_temp = temp[:-2*n]
test_temp = temp[-2*n:]
test_temp_2018 = temp[-n:]
# print(f"temp: {len(temp)}")
# print(f"train_temp: {len(train_temp)}")
# print(f"test_temp: {len(test_temp)}")


# # without weight
M1 = 2
trainN1 = 100
w1 = np.ones(trainN1)
model1 = PredictionModel(M1)
XP1 = model1.prediction(temp[-n-trainN1:], trainN1, w1, 365)
print(f"XP1 mse: {mean_square_error(test_temp_2018, XP1)}")
plt.plot(test_temp_2018, label="test")
plt.plot(XP1, label="XP1")
plt.legend()
plt.figure()

# # with weight
# gassian
M2 = 2
trainN2 = 100
mean = 350
var = 40
w2 = norm_weight_array(mean, var, trainN2)
model2 = PredictionModel(M2)
XP2 = model2.prediction(temp[-n-trainN2:], trainN2, w2, 365)
print(f"XP2 mse: {mean_square_error(test_temp_2018, XP2)}")
plt.plot(test_temp_2018, label="test")
plt.plot(XP2, label="XP2")
plt.legend()
plt.figure()

plt.plot(w2)
plt.show()

# M3 = 2
# trainN3 = 100
# slope = 0.000001
# bias = 0.0001
# # 0.000001 0.0001
# w3 = linear_weight_array(slope, bias, trainN3)
# model3 = PredictionModel(M3)
# XP3 = model3.prediction(temp[-n-trainN3:], trainN3, w3, 365)
# print(f"XP3 mse: {mean_square_error(test_temp_2018, XP3)}")
# plt.plot(test_temp_2018, label="test")
# plt.plot(XP3, label="XP3")
# plt.legend()
# plt.figure()

# M4 = 2
# trainN4 = 100
# low = 0.0001
# high = 0.0004
# t = 10
# # 0.0001 0.0004 10
# w4 = weight_array(low, high, t, trainN4)
# model4 = PredictionModel(M4)
# XP4 = model4.prediction(temp[-n-trainN4:], trainN4, w4, 365)
# print(f"XP4 mse: {mean_square_error(test_temp_2018, XP4)}")
# plt.plot(test_temp_2018, label="test")
# plt.plot(XP4, label="XP4")
# plt.legend()
# plt.figure()

# plt.plot(w2, label="w2")
# plt.plot(w3, label="w3")
# plt.plot(w4, label="w4")
# plt.legend()

# plt.show()

# 350.0, 40.0

# M1 = 2
# trainN1 = 100

# XPs = []
# mean = np.linspace(0, 350, 36)
# var = np.linspace(10, 200, 20)
# for i in mean:
#     for j in var:
#         w3 = norm_weight_array(i, j, trainN1)
#         model2 = PredictionModel(M1)
#         XP2 = model2.prediction(test_temp[-n-trainN1:], trainN1, w3, 365)
#         print("XPn mse: {0}".format(mean_square_error(
#             test_temp_2018, XP2)))
#         XPs.append([i, j, mean_square_error(
#             test_temp_2018, XP2)])
# print(XPs)

# mind = XPs[0]
# min = XPs[0][2]
# for i in XPs:
#     if i[2] < min:
#         min = i[2]
#         mind = i
# print(mind)


# plt.plot(test_temp_2018, label="test")
# plt.plot(XP2, label="XP with weight")
# plt.xlabel("date (day)", fontsize=16)
# plt.ylabel("tempearture (C)", fontsize=16)
# plt.legend()

# 窮舉
# M1 = [0, 1, 2, 3, 4, 5]
# trainN1 = [50, 100, 150, 200, 250, 300]

# XPs = []
# mses = []
# for i in M1:
#     for j in trainN1:
#         w1 = np.ones(j)
#         model1 = PredictionModel(i)
#         XP1 = model1.prediction(temp[-n-j:], j, w1, 365)
#         XPs.append(XP1)
#         print(f"M1: {i}, trainN1: {j}")
#         print("XP mse: {0}".format(mean_square_error(
#             test_temp_2018, XP1)))
#         mses.append(mean_square_error(
#             test_temp_2018, XP1))
# idx = 0
# for k in range(len(M1)):
#     fig, ax = plt.subplots(2, 3)
#     for i in range(2):
#         for j in range(3):
#             ax[i][j].plot(test_temp_2018)
#             ax[i][j].plot(XPs[idx])
#             ax[i][j].set_title(
#                 f"M: {k}, t: {trainN1[idx%6]}, mse: {round(mses[idx], 2)}")
#             idx += 1
plt.show()
