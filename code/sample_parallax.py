"""
This file is part of the TGASplus project.
Copyright 2016 David W. Hogg (NYU).
"""
import numpy as np

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
