"""
This file is part of the TGASplus project.
Copyright 2016 David W. Hogg (NYU).
"""
import numpy as np

def ln_oned_Gaussian(xs, mu, V):
    return -0.5 * (xs - mu) ** 2 / V - 0.5 * np.log(2. * np.pi * V)

class mixture_of_oned_Gaussians:

    def __init__(self, lnamps, means, vars):
        self.Q = len(lnamps)
        self.set_lnamps(lnamps)
        self.set_means(means)
        self.set_vars(vars)

    def set_lnamps(self, lnamps):
        self.lnamps = np.atleast_1d(lnamps)
        assert self.lnamps.shape == (self.Q, )

    def set_means(self, means):
        self.means = np.atleast_1d(means)
        assert self.means.shape == (self.Q, )

    def set_vars(self, vars):
        self.vars = np.atleast_1d(vars)
        assert self.vars.shape == (self.Q, )

    def set_pars(self, pars):
        self.set_lnamps( pars[0::3])
        self.set_means(pars[1::3])
        self.set_vars( pars[2::3])

    def get_pars(self):
        pars = np.zeros(3 * self.Q)
        pars[0::3] = self.lnamps
        pars[1::3] = self.means
        pars[2::3] = self.vars
        return pars

    def evaluate_ln(self, xs):
        """
        Not underflow-safe.
        """
        vals = -np.inf
        for q in range(self.Q):
            vals = np.logaddexp(vals, lnamps[q] + ln_oned_Gaussian(xs, self.means[q], self.vars[q]))
        return vals

if __name__ == "__main__":
    import pylab as plt
    lnamps = np.log([1.1, 0.5, 0.75])
    means = [1.2, 1.6, 2.1]
    vars = [0.01, 0.01, 0.01]
    foo = mixture_of_oned_Gaussians(lnamps, means, vars)
    xs = np.arange(0.005, 3.0, 0.01)
    plt.clf()
    plt.plot(xs, np.exp(foo.evaluate_ln(xs)), "k-")
    plt.savefig("deleteme.png")
