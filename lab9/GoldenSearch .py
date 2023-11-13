import matplotlib.pyplot as plt
import numpy as np
import time

a = np.array([2, -6, 2])  # y=2*(x-1.5)^2-2.5 [2, -6, 2]
f = np.poly1d(a)
dp = f.deriv()

x0 = -10
x1 = 15
threshold = 0.001


def golden_search(f, x0, x1, threshold):
    interval = []
    e = (-1+5**0.5)/2
    x2 = x0+(x1-x0)/(1+e)
    x3 = x0+(x2-x0)/(1+e)
    n = 0
    while abs(x0-x1) > threshold:
        interval.append((x0, x1, n))
        if f(x2) < f(x3):
            x0 = x3
            x3 = x2
            x2 = x1-(x1-x3)/(1+e)
        else:
            x1 = x2
            x2 = x3
            x3 = x0+(x2-x0)/(1+e)
        n += 1
    interval.append((x0, x1, n))
    x = (x1+x0)/2
    y = f(x)
    return x, y, interval

start = time.time()
x, y, interval = golden_search(f, x0, x1, threshold)
print(f"x: {x}\n y: {y}")
#print(interval)
end = time.time()
print(f"time: {end-start}")

# observe the interval
interval1x = [interval[0][0], interval[0][1]] 
interval1y = [f(interval[0][0]), f(interval[0][1])] 
interval2x = [interval[3][0], interval[3][1]] 
interval2y = [f(interval[3][0]), f(interval[3][1])] 
interval3x = [interval[10][0], interval[10][1]] 
interval3y = [f(interval[10][0]), f(interval[10][1])] 

plt.plot(np.linspace(-30, 30, 61), f(np.linspace(-30, 30, 61)))
plt.scatter(interval1x, interval1y, color="yellow", label="iter 0 interval")
plt.scatter(interval2x, interval2y, color="green", label="iter 3 interval")
plt.scatter(interval3x, interval3y, color="blue", label="iter 10 interval")
plt.title("golden search")
plt.legend()
plt.show()
