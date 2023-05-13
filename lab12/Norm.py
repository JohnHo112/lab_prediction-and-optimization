import numpy as np

x = np.array([1, 0, -2, 0, 3])


def norm(x, a):
    if a == 0:
        return len([i for i in x if i != 0])
    elif a == np.inf:
        return max(abs(x))
    elif a == -np.inf:
        return min(abs(x))
    else:
        return (sum(abs(x)**a))**(1/a)


print(f"zero norm: {norm(x, 0)}")
print(f"1-norm: {norm(x, 1)}")
print(f"2-norm: {norm(x, 2)}")
print(f"3-norm: {norm(x, 3)}")
print(f"Infinity norm: {norm(x, np.inf)}")
print(f"-Infinity norm: {norm(x, -np.inf)}")
