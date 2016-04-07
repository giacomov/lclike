__author__ = 'giacomov'

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_fit_results(time_resolved_results, redshift, triggername, decay_function):

    fig = plt.figure()

    # redshift correction /(1+args.redshif)

    tstarts = time_resolved_results["tstart"]
    tstops = time_resolved_results["tstop"]

    median = (tstarts + tstops) / 2.0 / (1 + float(redshift))
    start = tstarts / (1 + float(redshift))
    stop = tstops / (1 + float(redshift))

    y = time_resolved_results["photonFlux"]
    Dy = time_resolved_results["photonFluxError"]

    # Remove the "<" sign in the upper limits entries

    try:

        y = np.core.defchararray.replace(y, "<", "", count=None)

    except:

        print('No Upper-Limits Found in %s.' % (triggername))

    # Remove the "n.a." from the error column in the cases where there are upper limits,
    # and replace it with 0

    try:

        Dy = np.core.defchararray.replace(Dy, "n.a.", "0",
                                          count=None)
    except:

        print('No 0-Error Found in %s.' % (triggername))

    bar = 0.5
    color = "blue"

    Y = np.array(y, dtype=float)

    DY = np.array(Dy, dtype=float)

    plt.clf()

    if (DY > 0).sum() > 0:  # if sum() gives a non-zero value then there are error values

        plt.errorbar(median[DY > 0], Y[DY > 0],
                     xerr=[median[DY > 0] - start[DY > 0], stop[DY > 0] - median[DY > 0]],
                     yerr=DY[DY > 0], ls='None', marker='o', mfc=color, mec=color, ecolor=color, lw=2, label=None)

    if (DY == 0).sum() > 0:

        plt.errorbar(median[DY == 0], Y[DY == 0],
                     xerr=[median[DY == 0] - start[DY == 0], stop[DY == 0] - median[DY == 0]],
                     yerr=[bar * Y[DY == 0], 0.0 * Y[DY == 0]], lolims=True, ls='None', marker='', mfc=color, mec=color,
                     ecolor=color, lw=2, label=None)

    # Now plot the function

    tt = np.logspace(-3, np.log10(stop.max()), 200)
    fluxes = decay_function.getDifferentialFlux(tt)

    plt.loglog(tt / (1.0 + redshift), fluxes)

    plt.ylim([Y.min() / 10, Y.max()*10])
    plt.xlim(start.min(), stop.max())

    plt.suptitle('%s photonFlux per Time' % (triggername))
    plt.xlabel('Rest Frame Time(s)')
    plt.ylabel('Photon Flux')
    plt.xscale('symlog')
    plt.yscale('log')
    plt.grid(True)

    return fig