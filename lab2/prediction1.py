import numpy as np
import matplotlib.pyplot as plt


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


def linear_prediction(ak, X):
    n = len(X)
    L = len(ak)
    xp = 0

    for i in range(L):
        xp += ak[i]*X[n-1-i]
    return xp


def findOptimalValue(X, L):
    nL = len(X)
    A = np.zeros((L, L))
    B = np.zeros(L)

    for i in range(L):
        for j in range(L):
            x = 0
            for k in range(L, nL):
                x += X[k-1-i] * X[k-1-j]
            A[i][j] = x

    for i in range(L):
        x = 0
        for k in range(L, nL):
            x += X[k-i-1]*X[k]
        B[i] = x
    ak = np.linalg.inv(A).dot(B)
    return ak


def mean_square_error(X, XP, L):
    error = 0
    n = len(X)
    for i in range(n):
        error += (X[i]-XP[i])**2
    error = error/L
    return error


def prediction_a_year(ak, data, n):
    L = len(ak)
    XP = np.array([])
    for i in range(n):
        XP = np.append(XP, linear_prediction(ak, data[i:i+L]))
    return XP


filename = "temp1.txt"
date, temp = readfile(filename)
temp = np.array(temp)
temp = F2C(temp)

# temp
# train(2013): 0:365
# test(2017): 365:730
# test(2018): 730:1095

train_temp = temp[0:365]
test_temp_2017 = temp[365:730]
test_temp_2018 = temp[730:]


L = 12
n = 365  # number of prediction


ak = findOptimalValue(train_temp, L)
X = temp[730-L:]
XP = prediction_a_year(ak, X, n)

print("mse: {0}".format(mean_square_error(
    test_temp_2018, XP, n)))


# plot train data
plt.plot(train_temp, label="train")
plt.xlabel("date (day)", fontsize=16)
plt.ylabel("tempearture (C)", fontsize=16)
plt.legend()
plt.figure()

# plot test data and prediction
plt.plot(test_temp_2018, label="test")
plt.plot(XP, label="prediction")
plt.xlabel("date (day)", fontsize=16)
plt.ylabel("tempearture (C)", fontsize=16)
plt.legend()

plt.show()
