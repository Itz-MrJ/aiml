
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv('student.csv')
print(data.shape)
data.head()

math = data['Math'].values
read = data['Reading'].values
write = data['Writing'].values
print(math)



fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(math, read, write, color='r')
plt.legend()
plt.show()


m = len(math)
x0 = np.ones(m)
X = np.array([x0, math, read]).T



B = np.array([0, 0, 0])
Y = np.array(write)
print(Y)
alpha = 0.0001
def cost_function(X, Y, B):
    m = len(Y)
    J = np.sum((X.dot(B) - Y) ** 2)/(2 * m)
    return J

inital_cost = cost_function(X, Y, B)
print("Initial Cost")
print(inital_cost)

def gradient_descent(X, Y, B, alpha, iterations):
    cost_history = [0] * iterations
    m = len(Y)
    for iteration in range(iterations):
        h = X.dot(B)
        loss = h - Y
        gradient = X.T.dot(loss) / m
        B = B - alpha * gradient
    cost = cost_function(X, Y, B)
    cost_history[iteration] = cost
    return B, cost_history

newB, cost_history = gradient_descent(X, Y, B, alpha, 100000)
print("New Coefficients")
print(newB)
# Final Cost of new B
print("Final Cost")
print(cost_history[-1])

def r2_score(Y, Y_pred):
    mean_y = np.mean(Y)
    ss_tot = sum((Y - mean_y) ** 2)
    ss_res = sum((Y - Y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2


Y_pred = X.dot(newB)

print("R2 Score")
print(r2_score(Y, Y_pred))
