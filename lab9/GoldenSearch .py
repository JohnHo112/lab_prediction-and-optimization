import matplotlib.pyplot as plt
import numpy as np

a = np.array([2, -6, 2])  # y=2*(x-1.5)^2-2.5 [2, -6, 2]
f = np.poly1d(a)
dp = f.deriv()

x0 = -5
x1 = 5
threshold = 0.001


def golden_search(f, x0, x1, threshold):
    e = (-1+5**0.5)/2
    x2 = x0+(x1-x0)/(1+e)
    x3 = x0+(x2-x0)/(1+e)
    n = 1
    while abs(x0-x1) > threshold:
        if f(x2) < f(x3):
            x0 = x3
            x3 = x2
            x2 = x1-(x1-x3)/(1+e)
        else:
            x1 = x2
            x2 = x3
            x3 = x0+(x2-x0)/(1+e)
    x = (x1+x0)/2
    y = f(x)
    return x, y


x, y = golden_search(f, x0, x1, threshold)
print(x, y)

plt.plot(np.linspace(-30, 30, 61), f(np.linspace(-30, 30, 61)))
plt.show()
