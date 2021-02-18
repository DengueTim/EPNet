import os
import os.path
import logging

import numpy as np
from scipy.stats import norm
from scipy.optimize import curve_fit

def logger(name):
    logger = logging.getLogger(name)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s %(filename)s:%(lineno)s %(threadName)s] %(levelname)-8s %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S',
        )
    )
    logger.addHandler(stream_handler)
    logger.setLevel(logging.DEBUG)
    return logger

    #
    # file_handle_name = "file"
    # if file_handle_name in [h.name for h in logger.handlers]:
    #     return
    # if os.path.dirname(filepath) is not '':
    #     if not os.path.isdir(os.path.dirname(filepath)):
    #         os.makedirs(os.path.dirname(filepath))
    # file_handle = logging.FileHandler(filename=filepath, mode="a")
    # file_handle.set_name(file_handle_name)
    # file_handle.setFormatter(file_formatter)
    # logger.addHandler(file_handle)


class LossMonitor():
    def __init__(self):
        self.last = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, loss):
        self.last = loss
        self.sum += loss
        self.count +=1

    def average(self):
        return self.sum / self.count

def getImageFilenamesWithPaths(dir, filename_extention='.png'):
    filenames = [filename for filename in os.listdir(dir) if filename.endswith(filename_extention)]
    return [os.path.join(dir, filename) for filename in filenames]

def mult_gaussFun_Fit(xy,*m):
    A,x0,y0,varx,vary,rho,alpha = m
    X,Y = np.meshgrid(xy[0],xy[1])
    assert rho != 1
    a = 1/(2*(1-rho**2))
    Z = A*np.exp(-a*((X-x0)**2/(varx)+(Y-y0)**2/(vary)-(2*rho/(np.sqrt(varx*vary)))*(X-x0)*(Y-y0)))
    return Z.ravel()

def fit2dGuassian(data2d, mx=None, my=None):
    # Initial Guess
    l = np.size(data2d,0)
    d = data2d - np.amin(data2d)
    d /= np.sum(d)

    bin_centers = np.linspace(0.5, l - 0.5, l)
    # Initial Guess
    p0 = (np.amax(d), mx, my, 1, 1, 0.5, np.pi / 4)
    bounds = (
        [0, mx-0.1, my-0.1, 0, 0, 0.1, np.pi / 5],
        [np.amax(d), mx+0.1, my+0.1, 10, 10, 1, np.pi / 3])

    # Curve Fit parameters
    coeff, var_matrix = curve_fit(mult_gaussFun_Fit, (bin_centers, bin_centers), d.ravel(), p0=p0, bounds=bounds, verbose=1)

    return coeff, var_matrix
    # l = np.size(data2d,0)
    # min = np.amin(data2d)
    # d = data2d - min
    # d /= np.sum(d)
    #
    # (yis,xis) = np.mgrid[0:l,0:l]
    # px = np.multiply(xis, d)
    # py = np.multiply(yis, d)
    # if mx is None:
    #     mx = np.sum(px)
    # if my is None:
    #     my = np.sum(py)
    #
    # px = np.multiply(xis - mx, d)
    # py = np.multiply(yis - my, d)
    #
    # a = np.sqrt(np.sum(np.power(px, 2))) * l
    # b = np.sum(np.multiply(px, py)) * l
    # #b = np.sqrt(b) if b >=0 else -np.sqrt(-b)
    # c = np.sqrt(np.sum(np.power(py, 2))) * l
    #
    # return mx, my, np.array([[a,b],[b,c]])
