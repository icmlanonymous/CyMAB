import numpy as np
from numpy.random import binomial
from SMPyBandits.Arms import Bernoulli

class ExtendedBernoulli(Bernoulli):
    """ Extended Bernoulli distributed arm."""

    def __init__(self, probability, minr=-1.0, maxr=1.0):
        super(ExtendedBernoulli, self).__init__(probability)
        self.mean = probability * (maxr-minr) + minr  #: Mean for this Bernoulli arm
        self.minr = minr
        self.maxr = maxr
        self.ampl = maxr-minr

    # --- Random samples

    def draw(self, t=None):
        """ Draw one random sample."""
        return np.asarray(binomial(1, self.probability) * self.ampl + self.minr, dtype=float)

    def draw_nparray(self, shape=(1,)):
        """ Draw a numpy array of random samples, of a certain shape."""
        return np.asarray(binomial(1, self.probability, shape) * self.ampl + self.minr, dtype=float)