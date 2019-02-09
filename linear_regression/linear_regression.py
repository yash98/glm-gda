import sys
import statistics
import numpy as np
import matplotlib.pyplot as plt

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

theta = 0

def load_data (file_x, file_y):
    global x, y
    reader_x = open(file_x, "rt").read().splitlines()
    reader_y = open(file_y, "rt").read().splitlines()
    x = np.array(reader_x, dtype=float)
    y = np.array(reader_y, dtype=float)

def normalize_stats_data():
    global xn, x_mean, x_sd, x_min, x_max, xn_min, xn_max
    x_mean = statistics.mean(x)
    x_sd = statistics.stdev(x)
    xn = np.array([(x[i]-x_mean)/x_sd for i in range(x.size)])

    x_min = min(x)
    x_max = max(x)
    xn_min = (x_min-x_mean)/x_sd
    xn_max = (x_max-x_mean)/x_sd

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


def init():
    global theta
    load_data(sys.argv[1], sys.argv[2])
    # 1D data assumed
    theta = np.zeros(shape=(2), dtype=float)

def main():
    init()
    normalize_stats_data()
    batch_grad_desc(0.1, 0.000001)
    plot_hypothesis()

if (__name__=="__main__"):
    main()