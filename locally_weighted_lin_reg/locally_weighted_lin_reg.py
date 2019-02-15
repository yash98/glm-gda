import sys
import statistics
import math
import numpy as np
import matplotlib.pyplot as plt

x = 0
y = 0
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

theta = 0
theta_store = []

w = 0

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

def make_w(index, tau, belongs):
    global w
    w = np.zeros(shape=(x.size, x.size))
    for i in range(x.size):
        if (belongs):
            w[i][i] = math.exp(-((x[i]-x[index])**2)/(2*(tau**2)))
        else:
            w[i][i] = math.exp(-((x[i]-index)**2)/(2*(tau**2)))


def batch_grad_desc (eta, epsilon):
    global x, y, theta, theta_store

    # intialize
    # make totalError > epsilon
    lms_error = 10000000000.0
    prev_lms_error = lms_error + 1.0
    num_iterations = 0

    print("number of iterations: mean square error")
    while (lms_error>epsilon and num_iterations<100 and prev_lms_error >= lms_error+epsilon):
        prev_lms_error = lms_error
        lms_error = 0.0
        sumof_product_diff_x = np.zeros(2)
        for i in range(y.size):
            diff_actual_pred = y[i] - xn[i]*theta[1] - theta[0]
            sumof_product_diff_x[1] += diff_actual_pred*xn[i]
            sumof_product_diff_x[0] += diff_actual_pred
            lms_error += diff_actual_pred**2
        lms_error /= 2*y.size
        # update
        if(prev_lms_error>=lms_error):
            theta = theta + (eta*sumof_product_diff_x)/y.size
            theta_store.append(theta)
            if (num_iterations%10==0):
                print(str(num_iterations) + ": " + str(lms_error))
            num_iterations += 1
    
    print(str(num_iterations) + ": " + str(lms_error))
            
def plot_hypothesis_lin():
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.plot(x, y, marker='.', linestyle='None')
    ax.plot([x_min, x_max], [theta[0]+theta[1]*xn_min, theta[0]+theta[1]*xn_max], marker='None', label="hypothesis")
    plt.legend()
    plt.show()

def plot_hypothesis_local(tau):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.plot(x, y, marker='.', linestyle='None')
    for i in range(1000):
        x_start = x_min + i*(x_max-x_min)/1000
        x_end = x_min + (i+1)*(x_max-x_min)/1000

        make_w((x_start+x_end)/2, tau, False)
        weighted_normal_eq_solve()

        ax.plot([x_start, x_end], [theta[1][0]+theta[0][0]*x_start, theta[1][0]+theta[0][0]*x_end], marker='None', c='red')
    plt.legend()
    plt.show()

def weighted_normal_eq_solve():
    global theta, x_w_intercepts
    x_w_intercepts = np.array([[x[i], 1] for i in range(x.size)])
    xtw = np.matmul(np.transpose(x_w_intercepts), w)
    y_proper = y[np.newaxis].T
    coeff_y = np.matmul(np.linalg.inv(np.matmul(xtw, x_w_intercepts)),xtw)
    theta = np.matmul(coeff_y, y_proper)

def init():
    global theta, theta_store
    load_data(sys.argv[1], sys.argv[2])
    # 1D data assumed
    theta = np.zeros(shape=(2), dtype=float)
    theta_store.append(theta)

def main():
    init()
    normalize_stats_data()

    batch_grad_desc(0.4, 0.000001)
    plot_hypothesis_lin()

    plot_hypothesis_local(float(sys.argv[3]))

if (__name__=="__main__"):
    main()