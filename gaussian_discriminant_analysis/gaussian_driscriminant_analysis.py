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
x_mean = 0
x_sd = 0
x_min = 0
x_max = 0
x0 = 0
x1 = 0
phi = 0.0
mu = np.zeros(shape=(2, 2))
sigma = np.zeros(shape=(3,2,2))

def load_data (file_x, file_y):
    global x, y, y_int
    reader_x1 = list(csv.reader(open(file_x, "rt"), delimiter =' '))
    reader_x = []
    for i in range(len(reader_x1)):
        l = []
        for j in range(len(reader_x1[0])):
            if (reader_x1[i][j]!=''):
                l.append(reader_x1[i][j])
        reader_x.append(l)
    reader_y1 = open(file_y, "rt").read().splitlines()
    reader_y = [0 if reader_y1[i]=="Alaska" else 1 for i in range(len(reader_y1))]
    x = np.array(reader_x, dtype=float)
    y = np.array(reader_y, dtype=float)
    y_int = np.array(reader_y, dtype=int)

def stats_data():
    global x_mean, x_sd, x_min, x_max, x0, x1
    x0 = x[:,0]
    x1 = x[:,1]
    x_mean = [statistics.mean(x0), statistics.mean(x1)]
    x_sd = [statistics.stdev(x0), statistics.mean(x1)]

    x_min = [min(x0), min(x1)]
    x_max = [max(x0), max(x1)]

def set_phi():
    global phi
    for i in range(y_int.size):
        if y_int[i] == 1:
            phi += 1.0
    phi /= y_int.size

def set_means():
    global mu
    count = 0
    for i in range(y_int.size):
        if y_int[i] == 1:
            mu[1] += x[i]
            count += 1
        else:
            mu[0] += x[i]
    mu[0] /= (y_int.size-count)
    mu[1] /= count

def set_sigma():
    global sigma
    count = 0
    for i in range(y_int.size):
        if y_int[i] == 1:
            diff = x[i] - mu[1]
            diff = diff[np.newaxis]
            sigma[1] += np.matmul(diff.T,diff)
            count += 1
        else:
            diff = x[i] - mu[0]
            diff = diff[np.newaxis]
            sigma[0] += np.matmul(diff.T,diff)
    sigma[2] = (sigma[0]+sigma[1])/y_int.size
    sigma[0] /= (y_int.size-count)
    sigma[1] /= count

def plot_linear():
    plt.scatter(x0, x1, marker='.', linestyle='None', c=y_int)
    c = np.matmul(np.matmul(np.transpose(mu[1]), sigma[2]), mu[1])
    c -= np.matmul(np.matmul(np.transpose(mu[0]), sigma[2]), mu[0])
    cx = np.matmul(np.transpose(mu[0]-mu[1]), sigma[2])
    cxt = np.matmul(sigma[2], mu[0]-mu[1])
    cx0 = cxt[0]+cx[0]
    cx1 = cxt[1]+cx[1]

    axes = plt.gca()
    axes.set_ylim([x_min[1]-1.0,x_max[1]+1.0])
    plt.plot([x_min[0], x_max[0]], [-(cx0*x_min[0]+c)/cx1, -(cx0*x_max[0]+c)/cx1], marker='None')
    plt.show()

def init():
    load_data(sys.argv[1], sys.argv[2])
    stats_data()
    set_phi()
    set_means()
    set_sigma()

def main():
    init()
    plot_linear()

if (__name__=="__main__"):
    main()