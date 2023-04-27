# import matplotlib.pyplot as plt
# import numpy as np

# a = np.array([2, -6, 2])  # y=2*(x-1.5)^2-2.5 [2, -6, 2]
# f = np.poly1d(a)
# l = 0.5
# n = 1000
# x = -5


# def newton_method(x, f, l, n):
#     dp = f.deriv()
#     ddp = dp.deriv()

#     for _ in range(n):
#         x = x-l*dp(x)/ddp(x)

#     return x


# x = newton_method(x, f, l, n)
# y = f(x)
# print(x, y)


# plt.plot(np.linspace(-30, 30, 61), f(np.linspace(-30, 30, 61)))
# plt.show()

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d

# initial point
x0 = 1
y0 = 2

# set function
a = 0.5
b = 0.5
c = 10
x = np.array([a**2, 2*a, 1])
fx = np.poly1d(x)
y = np.array([b, 0, 0])
fy = np.poly1d(y)
f = fx(x0)+fy(y0)+c  # z=(0.5x+1)^2+0.5y^2+10

# set lambda to prevent nan and set iter number
l = 0.5
n = 50

# newton method


def newton_method(x0, y0, fx, fy, c, l, n):
    dfx = fx.deriv()
    ddfx = dfx.deriv()
    dfy = fy.deriv()
    ddfy = dfy.deriv()

    for _ in range(n):
        x0 = x0-l*(dfx(x0)/ddfx(x0))
        y0 = y0-l*(dfy(y0)/ddfy(y0))

    z = fx(x0)+fy(y0)+c
    return x0, y0, z


a, b, c = newton_method(x0, y0, fx, fy, c, l, n)
print(f"x_min: {a}\ny_min: {b}\nz_min: {c}")

# plot the function
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)
z = []
for i in x:
    t = []
    for j in y:
        t.append(fx(i)+fy(j)+c)
    z.append(t)
z = np.array(z)

ax.plot_surface(X, Y, z, cmap='rainbow')

plt.show()
