import numpy as np
import matplotlib.pyplot as plt

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


def mean(data):
    n = len(data)
    m = 0
    for i in data:
        m += i
    return m/n


def standard_deviation(data, mean):
    n = len(data)
    std = 0
    for i in range(n):
        std += (data[i]-mean)**2
    std = std/n
    return std**(1/2)


def covaricance(data1, data2, mean1, mean2):
    n = len(data1)
    cov = 0
    for i in range(n):
        cov += (data1[i]-mean1)*(data2[i]-mean2)
    return cov/n


def correlation(data1, data2):
    m1 = mean(data1)
    m2 = mean(data2)
    cov = covaricance(data1, data2, m1, m2)
    std1 = standard_deviation(data1, m1)
    std2 = standard_deviation(data2, m2)
    return cov/(std1*std2)


filename = "lab7/temp2.txt"
date, temp = readfile(filename)
temp = np.array(temp)
temp = F2C(temp)

n = 365

test_temp = temp[-2*n:]
test_temp_2017 = temp[-2*n:-n]
test_temp_2018 = temp[-n:]

d = np.array(np.linspace(1, 365, 365), int)
corrArr = np.array([])
for i in d:
    corr = correlation(test_temp[365-i:-i-1], test_temp_2018)
    corrArr = np.append(corrArr, corr)
    print(i, corr)

maxcorr = np.max(corrArr)
maxd = np.where(corrArr == maxcorr)
mincorr = np.min(corrArr)
mind = np.where(corrArr == mincorr)
print(f"max d: {maxd}\nmax corr: {maxcorr}")
print(f"min d: {mind}\nmin corr: {mincorr}")


plt.scatter(test_temp[365-(maxd[0][0]+1):-(maxd[0][0]+1)],
            test_temp_2018, label="max corr")
plt.legend()
plt.figure()
plt.scatter(test_temp[365-(mind[0][0]+1):-(mind[0][0]+1)],
            test_temp_2018, label="min corr")
plt.legend()
plt.figure()
plt.scatter(test_temp[365-91:-91], test_temp_2018, label="corr near 0")
plt.legend()
plt.figure()


# plot max corr and min corr
plt.plot(test_temp[365-(maxd[0][0]+1):-(maxd[0][0]+1)], label="max corr")
plt.plot(test_temp_2018, label="temp_2018")
plt.legend()
plt.figure()
plt.plot(test_temp[365-(mind[0][0]+1):-(mind[0][0]+1)], label="min corr")
plt.plot(test_temp_2018, label="temp_2018")
plt.legend()
plt.figure()
plt.plot(test_temp[365-91:-91], label="corr near 0")
plt.plot(test_temp_2018, label="temp_2018")
plt.legend()
plt.figure()

# plot corr curve
plt.plot(d, corrArr, label="corr curve")
plt.legend()
plt.show()

# print(correlation(test_temp[365-91:-91], test_temp_2018))
# print(np.corrcoef(test_temp[365-91:-91], test_temp_2018))
