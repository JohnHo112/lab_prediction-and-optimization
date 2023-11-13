import numpy as np
import matplotlib.pyplot as plt
import time


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


def mean_square_error(X, XP, L):
    error = 0
    n = len(X)
    for i in range(n):
        error += (X[i]-XP[i])**2
    error = error/L
    return error


def nonlinear_prediction(ak, bk, X):
    n = len(X)
    akL = len(ak)
    bkL = len(bk)
    xp = 0

    for i in range(akL):
        xp += ak[i]*X[n-1-i]

    for i in range(bkL):
        xp += bk[i]*((X[n-1-i]-X[n-1-i-1])**2)

    return xp


def findOptimalValue(X, L):
    nL = len(X)
    A = np.zeros((2*L-1, 2*L-1))
    B = np.zeros(2*L-1)

    # A
    for i in range(L):
        for j in range(L):
            x = 0
            for k in range(L, nL):
                x += X[k-1-i] * X[k-1-j]
            A[i][j] = x
    for i in range(L):
        for j in range(L-1):
            x = 0
            for k in range(L, nL):
                x += X[k-1-i] * (X[k-1-j] - X[k-1-j-1])**2
            A[i][j+L] = x
    for i in range(L-1):
        for j in range(L):
            x = 0
            for k in range(L, nL):
                x += ((X[k-1-i] - X[k-1-i-1])**2) * X[k-1-j]
            A[i+L][j] = x
    for i in range(L-1):
        for j in range(L-1):
            x = 0
            for k in range(L, nL):
                x += ((X[k-1-i] - X[k-1-i-1])**2) * (X[k-1-j] - X[k-1-j-1])**2
            A[i+L][j+L] = x

    # B
    for i in range(L):
        x = 0
        for k in range(L, nL):
            x += X[k-i-1]*X[k]
        B[i] = x
    for i in range(L-1):
        x = 0
        for k in range(L, nL):
            x += ((X[k-i-1]-X[k-i-1-1])**2)*X[k]
        B[i+L] = x

    optimalVal = np.linalg.inv(A).dot(B)
    return optimalVal[:L], optimalVal[L:]


def prediction_a_year(ak, bk, data, n):
    L = len(ak)
    XP = np.array([])
    for i in range(n):
        XP = np.append(XP, nonlinear_prediction(ak, bk, data[i:i+L]))
    return XP


filename = "lab3/temp1.txt"
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

L = 10
n = 365  # number of prediction

start = time.time()
ak, bk = findOptimalValue(train_temp, L)
# print(ak)
# print(bk)
X = temp[730-L:]
XP = prediction_a_year(ak, bk, X, n)
end = time.time()
print(f"time: {end-start}")

print("mse: {0}".format(mean_square_error(
    test_temp_2018, XP, n)))


# plot train data
plt.plot(train_temp, label="train")
plt.title("nonlinear model training data")
plt.xlabel("date (day)", fontsize=16)
plt.ylabel("tempearture (C)", fontsize=16)
plt.legend()
plt.figure()

# plot test data and prediction
plt.plot(test_temp_2018, label="test")
plt.title("nonlinear model testing data and prediction")
plt.plot(XP, label="prediction")
plt.xlabel("date (day)", fontsize=16)
plt.ylabel("tempearture (C)", fontsize=16)
plt.legend()


plt.show()
