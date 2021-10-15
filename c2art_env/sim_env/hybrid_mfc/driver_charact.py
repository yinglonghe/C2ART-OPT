import os, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm

pathRoot = os.path.abspath(__file__ + "/../../../../")
sys.path.append(pathRoot)

import c2art_env.utils as utils

class driver_charact():

    def __init__(self, id):
        self.id = id            
        self.driverType = ['dynamic', 'normal', 'timid'][id]
        self.shape = [0.410, 0.416, 0.335][id]
        self.location = [-0.089, -0.047, -0.081][id]
        self.scale = [0.405, 0.296, 0.295][id]

        v = np.linspace(0, 50, 101)
        fminLS = np.maximum(0.009 * v - 0.009, 0.021)
        fmaxLS = 3.20e-5 * v**3 - 0.003 * v**2 + 0.084 * v + 0.167
        self.fCurves = np.vstack((v, fminLS, fmaxLS))

        self.ids = None

    def gen_ids(self):
        while True:
            self.ids = lognorm.rvs(
                self.shape, 
                loc=self.location, 
                scale=self.scale
                )
            if self.ids >= 0 and self.ids <= 1:
                break

        return self.ids


    def get_ds(self, v, vn):
        if v < vn:
            fmin = utils.interp_binary(self.fCurves[0], self.fCurves[1], v)
            fmax = utils.interp_binary(self.fCurves[0], self.fCurves[2], v)

            ds = self.ids * (fmax - fmin) + fmin
        else:
            x, y = self.get_ids_pdf()
            ds = x[np.argmax(y)]

        return ds


    def get_ids_pdf(self):
        x = np.linspace(0, 1, 201)
        y = lognorm.pdf(x, self.shape, self.location, self.scale)

        return x, y


if __name__ == "__main__":
    # Plot IDS distributions for different drivers
    for id in range(3):
        model = driver_charact(id)
        sample = []
        x, y = model.get_ids_pdf()
        plt.plot(x, y, label=model.driverType+': PDF')
        for i in range(10000):
            if i % 1 == 0:
                ids = model.gen_ids()

            sample.append(ids)
            
        n, bins, patches = plt.hist(sample, 100, density=True, alpha=0.75,
            label=model.driverType+': Hist of a random sample')
    plt.xlabel('IDS')
    plt.ylabel('PDF')
    plt.grid()
    plt.legend()
    plt.show()

    # Plot fmin and fmax for speed-ds distribution
    plt.plot(model.fCurves[0], model.fCurves[2], 'g--', label='fmax')
    plt.plot(model.fCurves[0], model.fCurves[1], 'r--', label='fmin')
    plt.xlabel('speed (m/s)')
    plt.ylabel('DS')
    plt.grid()
    plt.legend()
    plt.show()