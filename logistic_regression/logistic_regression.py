import sys
import statistics
import math
import numpy as np
import csv
import matplotlib.pyplot as plt

x = 0
x0 = 0
x1 = 0
y = 0
y_int = 0
x_w_intercepts = 0
# normalization info
xn = 0
x_mean = 0
x_sd = 0
# x stats
x_min = 0
x_max = 0
xn_min = 0
xn_max = 0
# y stats
y_min = 0
y_max = 0

theta = 0
theta_store = []

def load_data (file_x, file_y):
    global x, y, y_int
    reader_x = list(csv.reader(open(file_x, "rt"), delimiter =','))
    reader_y = open(file_y, "rt").read().splitlines()
    x = np.array(reader_x, dtype=float)
    y = np.array(reader_y, dtype=float)
    y_int = np.array(reader_y, dtype=int)

def normalize_stats_data():
    global xn, x_mean, x_sd, x_min, x_max, xn_min, xn_max, x0, x1
    x0 = x[:,0]
    x1 = x[:,1]
    x_mean = [statistics.mean(x0), statistics.mean(x1)]
    x_sd = [statistics.stdev(x0), statistics.mean(x1)]
    xn = np.array([[(x[i][0]-x_mean[0])/x_sd[0], (x[i][1]-x_mean[1])/x_sd[1]] for i in range(y.size)])

    x_min = [min(x0), min(x1)]
    x_max = [max(x0), max(x1)]
    xn_min = [(x_min[0]-x_mean[0])/x_sd[0], (x_min[1]-x_mean[1])/x_sd[1]]
    xn_max = [(x_max[0]-x_mean[0])/x_sd[0], (x_max[1]-x_mean[1])/x_sd[1]]

    # y_min = min(y)
    # y_max = max(y)

def logisitic(index, arr):
    dot_prod = sum(theta*arr[index])
    if (dot_prod<-700):
        return 0
    return 1/(1+math.exp(-dot_prod))

def derivative():
    d1 = []
    for j in range(2):
        sum_i_diff = 0.0
        for i in range(y.size):
            sum_i_diff += (y[i]-logisitic(i, xn))*xn[i][j]
        d1.append(sum_i_diff)
    return np.array(d1)

def hessian_inv():
    d2 = np.zeros(shape=(2,2))
    for j in range(2):
        for k in range(2):
            for i in range(y.size):
                d2[j][k] += (-1)*logisitic(i, xn)*(1-logisitic(i, xn))*xn[i][j]*xn[i][k]
    return np.linalg.inv(d2)

def newton_method(epsilon):
    global theta
    jump = 1000000000000000000000
    prev_jump = 10000000000000000000000
    num_iterations = 0
    theta_update = 0
    while(jump > epsilon and jump<prev_jump and num_iterations<100000000):
        theta_update = np.matmul(hessian_inv(), derivative())
        theta = theta - theta_update
        prev_jump = jump
        jump = theta[0]**2+theta[1]**2
        theta_store.append(theta)
        # if (num_iterations%100==0):
        print(str(num_iterations) + ": " + str(theta))
        num_iterations += 1
    print(str(num_iterations) + ": " + str(theta))

def plot_hypothesis():
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel("X0")
    ax.set_ylabel("X1")
    axes = plt.gca()
    axes.set_ylim([x_min[1]-1.0,x_max[1]+1.0])
    plt.scatter(x0, x1, marker='.', linestyle='None', c=y_int)
    plt.plot([x_min[0], x_max[0]], [(-(theta[0]/theta[1])*xn_min[0])*x_sd[1]+x_mean[1], -((theta[0]/theta[1])*xn_max[0])*x_sd[1]+x_mean[1]], marker='None', label="Seperator")
    plt.legend()
    plt.show()

def init():
    global theta, theta_store
    load_data(sys.argv[1], sys.argv[2])
    theta = np.zeros(shape=(2), dtype=float)
    theta_store.append(theta)

def main():
    init()
    normalize_stats_data()

    newton_method(0.001)
    plot_hypothesis()

if (__name__=="__main__"):
    main()