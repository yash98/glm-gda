import sys
import statistics
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

x = 0
y = 0
# normalization info
xn = 0
x_mean = 0
x_sd = 0
# x stats
x_min = 0
x_max = 0
xn_min = 0
xn_max = 0
# y_sum = 0
# xn_sum = 0

theta = 0

def load_data (file_x, file_y):
    global x, y
    reader_x = open(file_x, "rt").read().splitlines()
    reader_y = open(file_y, "rt").read().splitlines()
    x = np.array(reader_x, dtype=float)
    y = np.array(reader_y, dtype=float)

def normalize_stats_data():
    global xn, x_mean, x_sd, x_min, x_max, xn_min, xn_max, y_sum, xn_sum
    x_mean = statistics.mean(x)
    x_sd = statistics.stdev(x)
    xn = np.array([(x[i]-x_mean)/x_sd for i in range(x.size)])

    x_min = min(x)
    x_max = max(x)
    xn_min = (x_min-x_mean)/x_sd
    xn_max = (x_max-x_mean)/x_sd

    # y_sum = sum(y)
    # xn_sum = sum(xn)


def batch_grad_desc (eta, epsilon):
    global x, y, theta

    # intialize
    # make totalError > epsilon
    lms_error = epsilon + 1.0
    num_iterations = 0

    while (lms_error>epsilon and num_iterations<100):
        lms_error = 0.0
        sumof_product_diff_x = np.zeros(2)
        for i in range(y.size):
            diff_actual_pred = y[i] - xn[i]*theta[1] - theta[0]
            sumof_product_diff_x[1] += diff_actual_pred*xn[i]
            sumof_product_diff_x[0] += diff_actual_pred
            lms_error += diff_actual_pred**2
        lms_error /= 2*y.size
        # update
        theta = theta + (eta*sumof_product_diff_x)/y.size
        num_iterations += 1
        print(str(num_iterations) + ": " + str(lms_error))
            
def plot_hypothesis():
    plt.plot(x, y, marker='.', linestyle='None')
    plt.plot([x_min, x_max], [theta[0]+theta[1]*xn_min, theta[0]+theta[1]*xn_max], marker='None')
    plt.show()

def mesh_of_error_function(t0, t1):
    tz = np.zeros(shape=(int(t0.size/t0[0].size), t0[0].size))
    for j in range(t0[0].size):
        for k in range(int(t0.size/t0[0].size)):
            z = 0.0
            for i in range(y.size):
                z += (y[i] - x[i]*t1[j][k] - t0[j][k])**2
            tz[j][k] = z/(2*y.size)
    return tz

def plot_3d_error_mesh():
    t0 = np.linspace(-10,10,50)
    t1 = np.linspace(-10,10,50)
    T0, T1 = np.meshgrid(t0, t1)
    Z = mesh_of_error_function(T0, T1)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(T0, T1, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_title('3D mesh')
    plt.show()

def init():
    global theta
    load_data(sys.argv[1], sys.argv[2])
    # 1D data assumed
    theta = np.zeros(shape=(2), dtype=float)

def main():
    init()
    normalize_stats_data()
    batch_grad_desc(1.3, 0.000001)
    plot_hypothesis()
    plot_3d_error_mesh()

if (__name__=="__main__"):
    main()