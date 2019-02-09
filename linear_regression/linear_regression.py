import csv
import sys
import numpy as np
import matplotlib.pyplot as plt

x = 0
y = 0
theta = 0

def load_data (file_x, file_y):
    global x, y
    reader_x = csv.reader(open(file_x, "rt"), delimiter =',')
    reader_y = csv.reader(open(file_y, "rt"), delimiter ='\n')
    list_x = list(reader_x)
    list_y = list(reader_y)
    x = np.array(list_x, dtype=float)
    y = np.array(list_y, dtype=float)
    y = y.flatten()

def batch_grad_desc (eta, epsilon):
    global x, y, theta

    # intialize
    # make totalError > epsilon
    lms_error = epsilon + 1.0

    while (lms_error>epsilon):
        total_error = 0.0
        sumof_product_diff_x1 = np.zeros(x[0].size+1)
        for i in range(y.size):
            diff_actual_pred = y[i] - np.dot(x[i], theta[1:]) - theta[0]
            sumof_product_diff_x1[1:] = sumof_product_diff_x1[1:] + diff_actual_pred*x[i]
            sumof_product_diff_x1[0] = diff_actual_pred
            total_error += diff_actual_pred**2
        lms_error /= 2*y.size
        # update
        theta = theta - (eta/y.size)*sumof_product_diff_x1     
            
def plot_hypothesis():
    global x, y, theta
    x1s = [x[i][0] for i in range(y.size)]
    plt.plot(x1s, y, marker='.', linestyle='None')

    min_x1s = min(x1s)
    print(min_x1s)
    max_x1s = max(x1s)
    print(max_x1s)
    plt.plot([min_x1s, max_x1s], [theta[0]+theta[1]*min_x1s, theta[0]+theta[1]*max_x1s], marker='None')
    plt.show()


def init():
    global theta
    load_data(sys.argv[1], sys.argv[2])
    theta = np.zeros(shape=(x[0].size+1), dtype=float)

def main():
    init()
    batch_grad_desc(0.7, 0.0001)
    print(theta)
    plot_hypothesis()

if (__name__=="__main__"):
    main()