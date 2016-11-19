"""
This file is part of the TGASplus project.
Copyright 2016 David W. Hogg (NYU).

# Bugs:
- All priors hard-set for relevance to the TGAS problem.
"""
import numpy as np
from scipy.misc import logsumexp
ln2pi = np.log(2. * np.pi)

def ln_oned_Gaussian(xs, mu, lnV):
    return -0.5 * (xs - mu) ** 2 / np.exp(lnV) - 0.5 * (ln2pi + lnV)

class mixture_of_oned_Gaussians:

    def __init__(self, lnamps, means, lnvars):
        """
        ## Bugs:
        - Full of MAGIC.
        """
        self.Q = len(lnamps)
        self.set_lnamps(lnamps)
        self.set_means(means)
        self.set_lnvars(lnvars)
        self.minlnamp = -16. # MAGIC
        self.maxlnamp = 0. # MAGIC
        self.minmean = 20. # MAGIC
        self.maxmean = 0. # MAGIC
        self.minlnvar = np.log(1.e-4) # MAGIC
        self.maxlnvar = 8. # MAGIC

    def set_lnamps(self, lnamps):
        self.lnamps = np.atleast_1d(lnamps)
        self.totallnamp = logsumexp(self.lnamps)
        assert self.lnamps.shape == (self.Q, )

    def set_means(self, means):
        self.means = np.atleast_1d(means)
        assert self.means.shape == (self.Q, )

    def set_lnvars(self, lnvars):
        self.lnvars = np.atleast_1d(lnvars)
        assert self.lnvars.shape == (self.Q, )

    def set_pars(self, pars):
        self.set_lnamps( pars[0::3])
        self.set_means(pars[1::3])
        self.set_lnvars( pars[2::3])

    def get_pars(self):
        pars = np.zeros(3 * self.Q)
        pars[0::3] = self.lnamps
        pars[1::3] = self.means
        pars[2::3] = self.lnvars
        return pars

    def evaluate_ln_likelihood(self, xs):
        """
        Not underflow-safe.
        """
        vals = -np.inf
        for q in range(self.Q):
            vals = np.logaddexp(vals, lnamps[q] + ln_oned_Gaussian(xs, self.means[q], self.lnvars[q]))
        return vals - self.totallnamp

    def evaluate_ln_prior(self):
        if np.any(self.lnamp < self.minlnamp):
            return -np.Inf
        if np.any(self.lnamp > self.maxlnamp):
            return -np.Inf
        if np.any(self.means < self.minmean):
            return -np.Inf
        if np.any(self.means > self.maxmean):
            return -np.Inf
        if np.any(self.lnvars < self.minlnvar):
            return -np.Inf
        if np.any(self.lnvars > self.maxlnvar):
            return -np.Inf
        return -0.5 * self.totallnamp ** 2 # MAGIC; weak prior on unit total amplitude

class parallax_catalog:

    def __init__(self, varpis, ivars):
        self.K = len(varpis)
        self.varpis = varpis.copy()
        self.ivars = ivars.copy()
        assert self.varpis.shape == (self.K, )
        assert self.ivars.shape == (self.K, )
        self.samples = None

    def make_posterior_samples(self):
        pass

    def get_posterior_samples(self):
        if self.samples is None:
            self.make_posterior_samples()
        return self.samples

if __name__ == "__main__":
    lnamps = np.log([1.1, 0.5, 0.75])
    means = [1.2, 1.6, 2.1]
    lnvars = np.log([0.04, 0.02, 0.015])
    foo = mixture_of_oned_Gaussians(lnamps, means, lnvars)
    dx = 0.01
    xs = np.arange(0.5 * dx, 3.0, dx)
    vals = np.exp(foo.evaluate_ln(xs))
    plt.clf()
    plt.plot(xs, vals, "k-")
    plt.savefig("deleteme.png")
    print(np.sum(np.exp(lnamps)), np.sum(vals) * dx)
