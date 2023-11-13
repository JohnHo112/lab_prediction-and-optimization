import numpy as np
import matplotlib.pyplot as plt
import math


# b = np.array(
#     [
#         [2, 3, 3, 4, 5, 4, 5],
#         [2, 3, 3, 4, 5, 4, 5],
#         [2, 3, 3, 4, 5, 4, 5],
#     ]
# )
# test data 2
# y = np.array([2, 3, 3, 4, 5, 4, 5, 4, 6, 7, 8, 9, 6, 10])
# b = np.array(
#     [
#         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#         [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
#         [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
#     ]
# )


def F(x, b):
    return x[0] * b[0] + x[1] * b[1] + x[2] * b[2]


def E(y, b, x, a):
    n = len(y)
    e = 0
    for i in range(n):
        e += abs(y[i] - x[0] * b[0][i] - x[1] * b[1][i] - x[2] * b[2][i]) ** a
    return e


class Minimize:
    def __init__(self, y, b, x, a, lr, n):
        self.yn = len(y)
        self.dim_x = len(x)
        self.y = y
        self.b = b
        self.x = x
        self.a = a
        self.lr = lr
        self.n = n

    def sn(self, i):
        sn = 1
        if (
            self.y[i]
            - self.x[0] * self.b[0][i]
            - self.x[1] * self.b[1][i]
            - self.x[2] * self.b[2][i]
            > 0
        ):
            sn = 1
        elif (
            self.y[i]
            - self.x[0] * self.b[0][i]
            - self.x[1] * self.b[1][i]
            - self.x[2] * self.b[2][i]
            < 0
        ):
            sn = -1
        elif (
            self.y[i]
            - self.x[0] * self.b[0][i]
            - self.x[1] * self.b[1][i]
            - self.x[2] * self.b[2][i]
            == 0
        ):
            sn = 0
        return sn

    def gradient(self):
        g = np.zeros(self.dim_x)
        for i in range(self.dim_x):
            grad_e = 0
            for j in range(self.yn):
                grad_e += (
                    self.b[i][j]
                    * self.sn(j)
                    * abs(
                        self.y[j]
                        - self.x[0] * self.b[0][j]
                        - self.x[1] * self.b[1][j]
                        - self.x[2] * self.b[2][j]
                    )
                    ** (self.a - 1)
                )
            g[i] = -1 * self.a * grad_e
        return g

    def gradient_descent(self):
        self.norm_record = []
        self.x_record = []
        for i in range(self.n):
            g = self.gradient()
            self.x = self.x - self.lr * g
            # record x and norm
            self.x_record.append(self.x)
            self.norm_record.append(
                np.linalg.norm(
                    (
                        self.y
                        - self.x[0] * self.b[0]
                        - self.x[1] * self.b[1]
                        - self.x[2] * self.b[2]
                    ),
                    self.a,
                )
            )
        return self.norm_record, self.x_record


def plot_norm(y):
    plt.plot(y)
    plt.title("norm record")
    plt.xlabel("iteration number")
    plt.ylabel("error norm")
    plt.figure()


def plot_y_f(x, y, b, n):
    X = np.arange(0, len(y), 1)
    plt.scatter(X, y, label="y")
    plt.scatter(X, F(x[n], b), label=f"iter{n} function")
    plt.title("y and minimize function")
    plt.legend()
    plt.figure()


################ test data2 useful function ###############
def random_y(x, t):
    a = np.zeros(len(x))
    for i in range(len(x)):
        a[i] = np.random.randint(i - t, i + t)
    return a


def sin(x):
    return np.sin(x)


def linear(x, a, b):
    return a * x + b


############### set parameters ################
a = 1.5
lr = 0.0001
n = 1000

############### test data1 ###############
x = np.array([0, 0, 0])
y = np.array([2, 3, 3, 4, 5, 4, 5])
b = np.array(
    [
        [1, 1, 1, 1, 1, 1, 1],
        [1, 2, 3, 4, 5, 6, 7],
        [1, -1, 1, -1, 1, -1, 1],
    ]
)
model = Minimize(y, b, x, a, lr, n)
norm_record, x_record = model.gradient_descent()
plot_norm(norm_record)
plot_y_f(x_record, y, b, 0)
plot_y_f(x_record, y, b, 50)
plot_y_f(x_record, y, b, 999)

print("test data1")
print(f"iter 0 x: {x_record[0]}")
print(f"norm: {norm_record[0]}")
print(f"iter 50 x: {x_record[50]}")
print(f"norm: {norm_record[50]}")
print(f"iter 999 x: {x_record[999]}")
print(f"norm: {norm_record[999]}")
print("\n")
plt.show()


############### test data2 ###############
x = np.array([0, 0, 0])
yn = 50
X = np.arange(0, yn, 1)
Y = random_y(X, 20)
b1 = sin(X)
b2 = linear(X, 0.5, 0)
b3 = np.ones(yn)
b = np.array([b1, b2, b3])
model1 = Minimize(Y, b, x, a, lr, n)
norm_record1, x_record1 = model1.gradient_descent()

plot_norm(norm_record1)
plot_y_f(x_record1, Y, b, 0)
plot_y_f(x_record1, Y, b, 50)
plot_y_f(x_record1, Y, b, 999)

print("test data2")
print(f"iter 0 x: {x_record1[0]}")
print(f"norm: {norm_record1[0]}")
print(f"iter 50 x: {x_record1[50]}")
print(f"norm: {norm_record1[50]}")
print(f"iter 999 x: {x_record1[999]}")
print(f"norm: {norm_record1[999]}")
plt.show()
