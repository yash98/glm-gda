import csv
import numpy
import sys

x = 0
y = 0
theta = 0

def load_data (file_x, file_y):
    global x
    global y
    reader_x = csv.reader(open(file_x, "rt"), delimiter =',')
    reader_y = csv.reader(open(file_y, "rt"), delimiter =',')
    list_x = list(reader_x)
    list_y = list(reader_y)
    x = numpy.array(list_x, dtype=float)
    y = numpy.array(list_y, dtype=float)

def init():
    global theta
    load_data(sys.argv[1], sys.argv[2])
    theta = numpy.zeros(shape=(x.size, 1), dtype=float)

def main():
    init()

if (__name__=="__main__"):
    main()