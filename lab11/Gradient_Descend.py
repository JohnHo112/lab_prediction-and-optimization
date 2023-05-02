import matplotlib.pyplot as plt
import numpy as np
import time

# parameters for gradient descend
lr = 0.2
n = 100

# parameters for paraboloid
a = 1
b = 1
c = 0

# parameters for golden search
j = -5
k = 5


def f(x, y):  # set function
    # return a*x**2+b*y**2+c
    return a*(x-1)**2+b*(y+5)**2+c+x*y  # 14/3 -22/3 -46/3


def gradient(f, x, y, h=1e-6):  # gradient for 3 dim
    g = np.zeros(2)
    grad_x = (f(x+h, y)-f(x, y))/h
    grad_y = (f(x, y+h)-f(x, y))/h
    g[0] = grad_x
    g[1] = grad_y
    return g


# golden_search for finding learning rate
def golden_search(f, x0, x1, threshold=1e-6):
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


# c_plot = []


def gradient_descend(f, x0, y0, lr, n, split, threshold=1e-6):  # gradient_descend function
    for i in range(n):
        g = gradient(f, x0, y0)
        # print(f"iter: {i} norm: {np.linalg.norm(g)}")
        if np.linalg.norm(g) < threshold:
            break

        if split:
            # min c
            def tmp_f(c):
                return f(x0+c*g[0], y0+c*g[1])
            c, yc = golden_search(tmp_f, j, k, threshold=1e-6)
            # c_plot.append(c)
            x0 = x0+c*g[0]
            y0 = y0+c*g[1]

        else:
            # learning rate
            c = lr
            x0 = x0-c*g[0]
            y0 = y0-c*g[1]

    print(f"x_min: {x0}\ny_min: {y0}\nf_min: {f(x0, y0)}")
    print(f"iter num: {i}")


# set initial point
x0 = 1000
y0 = 10

# main function
print("min_c")
start = time.time()
gradient_descend(f, x0, y0, lr, n, True)
end = time.time()
print(f"min_c sec: {end-start}\n")

# print(c_plot)
# plt.plot(c_plot)
# plt.show()

print("fix_c")
start = time.time()
gradient_descend(f, x0, y0, lr, n, False)
end = time.time()
print(f"fix_c sec: {end-start}\n")

# # plot the function
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = np.arange(-50, 51)
y = np.arange(-50, 51)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

ax.plot_surface(X, Y, Z, cmap='rainbow')

plt.show()
