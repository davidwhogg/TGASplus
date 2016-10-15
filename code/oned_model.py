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

def sample_one_star_parallax(varpi, sigma, T):
    """
    ## bugs:
    - Prior maximum length hard-coded.
    - Units ugly: varpi required in mas; 1/varpi in kpc.
    - I don't like the samplings; as I increse magic_number,
      the fraction that survive the rejection goes down.
    """
    prior_length = 1. # kpc
    magic_number = 128 # MAGIC
    varpis = np.array([])

    iter = 1
    while len(varpis) < T:
        nstart = 2 * iter * magic_number
        print("iter {}; s/n {}".format(iter, varpi/sigma))

        # sample from the "likelihood" and reject using the prior
        foos = varpi + sigma * np.random.normal(size=nstart)
        distances = 1. / foos
        priors = np.zeros_like(foos)
        good = (foos > (1. / prior_length))
        if np.sum(good):
            priors[good] = foos[good] ** -4
            bars = np.random.uniform(0., np.max(priors), size=len(foos))
            accepts = priors > bars
            print("  generated {} likelihood trials and {} survived the prior".format(nstart, np.sum(accepts)))
            if np.sum(accepts) > magic_number:
                varpis = np.append(varpis, foos[accepts])

        # sample from the prior and reject using the likelihood
        foos = 1. / (np.random.uniform(0., prior_length ** 3, size=nstart) ** (1. / 3.)) # prior
        distances = 1. / foos
        likes = np.exp(-0.5 * (foos - varpi) ** 2 / sigma ** 2)
        bars = np.random.uniform(0., np.max(likes), size=len(foos))
        accepts = likes > bars
        print("  generated {} prior trials and {} survived the likelihood".format(nstart, np.sum(accepts)))
        if np.sum(accepts) > magic_number:
            varpis = np.append(varpis, foos[accepts])
        iter += 1

    return varpis[np.random.randint(len(varpis), size=T)]

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
    import pylab as plt
    sigma = 19.
    T = 1024
    vmax = 500.
    dv, dhv = 0.1, 2.
    varpiplot = np.arange(1. + 0.5 * dv, vmax + 0.5 * dv, dv)
    priorplot = varpiplot ** -4
    plotnum = 0
    for varpi in np.arange(450., 20., -5.):
        posteriorplot = priorplot * np.exp(-0.5 * (varpi - varpiplot) ** 2 / sigma ** 2)
        posteriorplot = posteriorplot / (np.sum(posteriorplot * dv))
        varpis = sample_one_star_parallax(varpi, sigma, T)
        plt.clf()
        bins = np.arange(-0.5*dhv, vmax + 0.5*dhv, dhv)
        plt.hist(varpis, bins=bins, histtype="step", color="r")
        plt.axvline(varpi, color="k")
        plt.axvline(varpi-sigma, color="k", alpha=0.5)
        plt.axvline(varpi+sigma, color="k", alpha=0.5)
        plt.plot(varpiplot, T * dhv * posteriorplot, "b-")
        plt.xlim(0., vmax)
        plt.xlabel("true parallax (mas)")
        plt.ylim(0., 100.)
        plt.text(varpi, 90., "measured parallax", rotation=90., ha="right")
        plt.savefig("varpis_{:02d}.png".format(plotnum))
        plotnum += 1

if False:
    lnamps = np.log([1.1, 0.5, 0.75])
    means = [1.2, 1.6, 2.1]
    vars = [0.04, 0.01, 0.01]
    foo = mixture_of_oned_Gaussians(lnamps, means, vars)
    dx = 0.01
    xs = np.arange(0.5 * dx, 3.0, dx)
    vals = np.exp(foo.evaluate_ln(xs))
    plt.clf()
    plt.plot(xs, vals, "k-")
    plt.savefig("deleteme.png")
    print(np.sum(np.exp(lnamps)), np.sum(vals) * dx)
