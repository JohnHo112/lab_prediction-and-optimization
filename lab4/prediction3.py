import numpy as np
import matplotlib.pyplot as plt
import math
import time

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


# create a weight array (yn: number of train year, lyn: number of leap year, s: number of weight array repeat)
def uniform_weight(low, high, yn, lyn, s):
    W = np.array([])
    for i in range(s):
        w = np.array([])
        a = np.ones(365)*low
        shiftIdx = int(365/s)
        for j in range(shiftIdx):
            a[j+i*shiftIdx] = high
        for k in range(yn):
            w = np.append(w, a)
        w = np.append(w, w[-lyn:])
        W = np.append(W, w)
    W = np.reshape(W, (s, (yn*365+lyn)))

    return W


def gaussian_weight(mean, sigma, yn, lyn, s):
    x = np.linspace(0, 364, 365)
    W = np.array([])
    for i in range(s):
        w = np.array([])
        shiftIdx = int(365/s)*i
        a = normal_distribution(x, mean+shiftIdx, sigma)
        for k in range(yn):
            w = np.append(w, a)
        w = np.append(w, w[-lyn:])
        W = np.append(W, w)
    W = np.reshape(W, (s, (yn*365+lyn)))

    return W


class PredictionModel:  # model
    def __init__(self, L, a, W):
        self.L = L
        self.a = a
        self.W = W
        self.Ak = np.array([])
        self.Bk = np.array([])

    def find_optimal_value(self, X, w):
        nL = len(X)
        A = np.zeros((2*self.L-1, 2*self.L-1))
        B = np.zeros(2*self.L-1)

        # A
        for i in range(self.L):
            for j in range(self.L):
                x = 0
                for k in range(self.L, nL):
                    x += w[k]*(X[k-1-i]*X[k-1-j])
                A[i][j] = x
        for i in range(self.L):
            for j in range(self.L-1):
                x = 0
                for k in range(self.L, nL):
                    # x += w[k]*X[k-1-i]*(X[k-1-j]-X[k-1-j-1])**a
                    x += w[k] * \
                        (X[k-1-i] * (abs(X[k-1-j] - X[k-1-j-1]))**self.a)
                A[i][j+self.L] = x
        for i in range(self.L-1):
            for j in range(self.L):
                x = 0
                for k in range(self.L, nL):
                    # x += w[k]*((X[k-1-i]-X[k-1-i-1])**a)*X[k-1-j]
                    x += w[k] * \
                        ((abs(X[k-1-i] - X[k-1-i-1])**self.a) * X[k-1-j])
                A[i+self.L][j] = x
        for i in range(self.L-1):
            for j in range(self.L-1):
                x = 0
                for k in range(self.L, nL):
                    # x += w[k]*((X[k-1-i]-X[k-1-i-1])**a)*((X[k-1-j]-X[k-1-j-1])**a)
                    x += w[k]*((abs(X[k-1-i] - X[k-1-i-1])**self.a)
                               * (abs(X[k-1-j] - X[k-1-j-1])**self.a))
                A[i+self.L][j+self.L] = x

        # B
        for i in range(self.L):
            x = 0
            for k in range(self.L, nL):
                x += w[k]*(X[k-i-1]*X[k])
            B[i] = x
        for i in range(self.L-1):
            x = 0
            for k in range(self.L, nL):
                # x += w[k]*((X[k-i-1]-X[k-i-1-1])**a)*X[k]
                x += w[k]*((abs(X[k-i-1]-X[k-i-1-1])**self.a)*X[k])
            B[i+self.L] = x

        optimalVal = np.linalg.inv(A).dot(B)
        return optimalVal[:self.L], optimalVal[self.L:]   # return ak and bk

    # using weight array to compute Ak and Bk
    def find_optimal_value_weight(self, X):
        for w in self.W:
            ak, bk = self.find_optimal_value(X, w)

            self.Ak = np.append(self.Ak, ak)
            self.Bk = np.append(self.Bk, bk)

        self.Ak = np.reshape(self.Ak, (self.W.shape[0], int(
            self.Ak.shape[0]/self.W.shape[0])))
        self.Bk = np.reshape(self.Bk, (self.W.shape[0], int(
            self.Bk.shape[0]/self.W.shape[0])))

    def nonlinear_prediction(self, ak, bk, X):
        n = len(X)
        akL = len(ak)
        bkL = len(bk)
        xp = 0

        for i in range(akL):
            xp += ak[i]*X[n-1-i]

        for i in range(bkL):
            # xp += bk[i]*((X[n-1-i]-X[n-1-i-1])**a)
            xp += bk[i]*(abs(X[n-1-i]-X[n-1-i-1])**self.a)

        return xp

    def prediction_a_year(self, data, n):
        XP = np.array([])
        s = self.W.shape[0]
        idx = 0

        for i in range(s):
            for j in range(int(n/s)):
                XP = np.append(XP, self.nonlinear_prediction(
                    self.Ak[i], self.Bk[i], data[idx:idx+self.L]))
                idx += 1

        return XP


filename = "lab4/temp2.txt"
date, temp = readfile(filename)
temp = np.array(temp)
temp = F2C(temp)

# set parameters
n = 365
L = 17
a = 2.2
s = 5

low = 0.1
high = 0.24
var = 250
mean = 45

# data
test_temp_2018 = temp[-n:]
train_temp = temp[:-2*n]
test_temp = temp[-2*n:]
X = temp[-n-L:]

# set weight array
W = uniform_weight(low, high, 10, 3, s)
# W = gaussian_weight(mean, var, 10, 3, s)

start = time.time()
model = PredictionModel(L, a, W)  # create prediction mode
model.find_optimal_value_weight(train_temp)  # find optimal value
XP = model.prediction_a_year(X, 364)  # predict a year
end = time.time()
print(f"time: {end-start}")

# print("L: {0}\na: {1}\nlow: {2}\nhigh: {3}\ns: {4}".format(L, a, low, high, s))
# print("L: {0}\na: {1}\nvar: {2}\nmean: {3}".format(L, a, var, mean))
print("mse: {0}".format(mean_square_error(
    test_temp_2018, XP)))


# plot train data
plt.plot(train_temp, label="train")
plt.title("generalized nonlinear prediction training data")
plt.xlabel("date (day)", fontsize=16)
plt.ylabel("tempearture (C)", fontsize=16)
plt.legend()
plt.figure()

plt.plot(W[0][:365], label="w1")
plt.plot(W[1][:365], label="w2")
plt.plot(W[2][:365], label="w3")
plt.plot(W[3][:365], label="w4")
plt.plot(W[4][:365], label="w5")
plt.title("weight")
plt.legend()
plt.figure()

# plot test data and prediction
plt.plot(test_temp_2018, label="test")
plt.plot(XP, label="prediction")
plt.title("generalized nonlinear prediction testing data and prediction")
plt.xlabel("date (day)", fontsize=16)
plt.ylabel("tempearture (C)", fontsize=16)
plt.legend()

plt.show()
