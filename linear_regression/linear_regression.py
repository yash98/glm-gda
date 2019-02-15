import sys
import statistics
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.animation import FuncAnimation

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
theta_store = []

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
    global x, y, theta, theta_store

    # intialize
    # make totalError > epsilon
    lms_error = epsilon + 1.0
    prev_lms_error = lms_error + 1.0
    num_iterations = 0

    print("number of iterations: mean square error")
    while (lms_error>epsilon and num_iterations<100 and prev_lms_error >= lms_error):
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
            
def plot_hypothesis():
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(x, y, marker='.', linestyle='None')
    ax.plot([x_min, x_max], [theta[0]+theta[1]*xn_min, theta[0]+theta[1]*xn_max], marker='None')
    plt.show()

def mesh_of_error_function(t0, t1):
    tz = np.zeros(shape=(int(t0.size/t0[0].size), t0[0].size))
    for j in range(t0[0].size):
        for k in range(int(t0.size/t0[0].size)):
            tz[j][k] = error_function(t0[j][k], t1[j][k])

    return tz

def error_function(t0s, t1s):
    z = 0.0
    for i in range(y.size):
        z += (y[i] - xn[i]*t1s - t0s)**2
    z /= (2*y.size)
    return z

t0first = 0
t0last = 0
t1first = 0
t1last = 0
T0 = 0
T1 = 0
Z = 0

def init_theta_3d_plotting():
    global t0first, t0last, t1first, t1last, T0, T1, Z
    t0first = theta_store[0][0]
    t0last = theta_store[len(theta_store)-1][0]
    t1first = theta_store[0][1]
    t1last = theta_store[len(theta_store)-1][1]
    # t0 = np.linspace(t0first, 2*t0last-t0first,70)
    # t1 = np.linspace(t1first, 2*t1last-t1first,70)
    t0 = np.linspace(-2.0, 2.0,70)
    t1 = np.linspace(-2.0, 2.0,70)
    T0, T1 = np.meshgrid(t0, t1)
    Z = mesh_of_error_function(T0, T1)

def plot_3d_error_mesh(time_gap):
    fig = plt.figure()
    ax = p3.Axes3D(fig)

    ax.set_title('3D mesh of error function')
    ax.set_xlabel('Theta0')
    ax.set_ylabel('Theta1')
    ax.set_zlabel('J(Theta)')

    ax.plot_surface(T0, T1, Z, rstride=1, cstride=1, cmap='plasma', edgecolor='none')

    for i in range(len(theta_store)):
        plt.pause(time_gap)
        ax.scatter3D([theta_store[i][0]], [theta_store[i][1]], [error_function(theta_store[i][0], theta_store[i][1])], c='g', marker='o')
    
    plt.show()

def plot_contours(time_gap):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    max_contour = 0.9*error_function(t0first, t1first)
    for i in range(6):
        contour_level = i*(max_contour/5)
        plt.contour(T0, T1, Z, [contour_level], colors='k')

    for i in range(len(theta_store)):
        plt.pause(time_gap)
        ax.plot([theta_store[i][0]], [theta_store[i][1]], c='g', marker='o')

    plt.show()

def init():
    global theta, theta_store
    load_data(sys.argv[1], sys.argv[2])
    # 1D data assumed
    theta = np.zeros(shape=(2), dtype=float)
    theta_store.append(theta)

def main():
    init()
    normalize_stats_data()
    batch_grad_desc(float(sys.argv[3]), 0.00001)
    plot_hypothesis()
    init_theta_3d_plotting()
    plot_3d_error_mesh(float(sys.argv[4]))
    plot_contours(float(sys.argv[4]))

if (__name__=="__main__"):
    main()