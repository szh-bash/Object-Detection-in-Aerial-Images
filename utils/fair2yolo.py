import os
import sys
import numpy as np
from math import sqrt, atan, acos, fabs
label_origin = '/home/shenzhonghai/Object-Detection-in-Aerial-Images/DOTA_demo_view/train_val/labelTxt/'
label_target = '/home/shenzhonghai/Object-Detection-in-Aerial-Images/DOTA_demo_view/train_val/labels/'
pi = acos(-1)


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def dist(a, b):
    return sqrt(a*a+b*b)


def get_ang(u):
    dx = x[(u+1) % 4]-x[u]
    dy = y[(u+1) % 4]-y[u]
    dy = -dy
    if dx < 0:
        dx = -dx
        dy = -dy
    # print(dx, dy)
    if dx == 0.0:
        return 0.5
    if dy > 0:
        return 1 - atan(dy/dx)/pi
    return atan(-dy/dx)/pi


if __name__ == "__main__":
    allDir = os.listdir(label_origin)
    # print(allDir)
    for file in allDir:
        # if not (file == '983__1__512___2048.txt'):
        #     continue
        # print(file)
        f = open(label_origin+file)
        st = f.readline().split(' ')
        # sys.stdout = Logger(label_target+file, sys.stdout)
        out = open(label_target+file, mode='w')
        while len(st)-(st[0] == ''):
            # print(st)
            x = np.array([st[a*2] for a in range(4)], dtype=np.float64)
            y = np.array([st[a*2+1] for a in range(4)], dtype=np.float64)

            # print(x[0]+x[2], x[1]+x[3])
            # print(y[0]+y[2], y[1]+y[3])
            # print(dist(x[1]-x[0], y[1]-y[0]))
            # print(dist(x[2] - x[1], y[2] - y[1]))
            # print(dist(x[3] - x[2], y[3] - y[2]))
            # print(dist(x[0] - x[3], y[0] - y[3]))
            xc = sum(x)/4
            yc = sum(y)/4
            w0 = dist(x[1] - x[0], y[1] - y[0])
            w1 = dist(x[3] - x[2], y[3] - y[2])
            W = (w0 + w1)/2
            h0 = dist(x[2] - x[1], y[2] - y[1])
            h1 = dist(x[0] - x[3], y[0] - y[3])
            H = (h0 + h1)/2
            long = max(W, H)
            short = min(W, H)
            theta = (get_ang(0)+get_ang(2))/2 if W > H else (get_ang(1)+get_ang(3))/2
            # print(w0, w1)
            # print(h0, h1)

            xc /= 1024
            yc /= 1024
            long /= 1024
            short /= 1024
            theta *= 180

            if fabs(w0-w1)<=2 and fabs(h0-h1)<=2:
                print(0, xc, yc, long, short, theta)
                out.write("%d %.8f %.8f %.8f %.8f %.8f\n" %(0, xc, yc, long, short, theta))
            st = f.readline().split(' ')
        out.close()
