__author__ = 'giacomov'

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_fit_results(time_resolved_results, redshift, triggername, decay_function=None, flux_type='photonFlux'):

    fig, (ax1, ax2) = plt.subplots(2,1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    # redshift correction /(1+args.redshif)

    tstarts = time_resolved_results["tstart"]
    tstops = time_resolved_results["tstop"]

    median = (tstarts + tstops) / 2.0 / (1 + float(redshift))
    start = tstarts / (1 + float(redshift))
    stop = tstops / (1 + float(redshift))

    y = time_resolved_results[flux_type]
    Dy = time_resolved_results[flux_type+"Error"]

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

    if (DY > 0).sum() > 0:  # if sum() gives a non-zero value then there are error values

        ax1.errorbar(median[DY > 0], Y[DY > 0],
                     xerr=[median[DY > 0] - start[DY > 0], stop[DY > 0] - median[DY > 0]],
                     yerr=DY[DY > 0], ls='None', marker='o', mfc=color, mec=color, ecolor=color, lw=2, label=None)

    if (DY == 0).sum() > 0:

        ax1.errorbar(median[DY == 0], Y[DY == 0],
                     xerr=[median[DY == 0] - start[DY == 0], stop[DY == 0] - median[DY == 0]],
                     yerr=[bar * Y[DY == 0], 0.0 * Y[DY == 0]], lolims=True, ls='None', marker='', mfc=color, mec=color,
                     ecolor=color, lw=2, label=None)

    # Now plot the function
    
    if decay_function is not None:
    
        tt = np.logspace(-3, np.log10(stop.max()), 200)
        fluxes = decay_function.getDifferentialFlux(tt)

        ax1.loglog(tt / (1.0 + redshift), fluxes)

    ax1.set_ylim([Y.min() / 10, Y.max()*10])
    ax1.set_xlim(start.min(), stop.max())

    plt.suptitle('%s' % (triggername))
    
    if redshift > 0:
    
        ax1.set_xlabel('Rest Frame Time(s)')
    
    else:
        
        ax1.set_xlabel('Time(s)')
    
    ax1.set_ylabel('Photon Flux')
    ax1.set_xscale('symlog')
    ax1.set_yscale('log')
    #plt.grid(True)
    
    ax2.errorbar(median[DY > 0], time_resolved_results['photonIndex'][DY>0], 
                 xerr=[median[DY > 0] - start[DY > 0], stop[DY > 0] - median[DY > 0]],
                 yerr=np.array(time_resolved_results['photonIndexError'][DY > 0], dtype=float),
                 fmt='.') 
    
    ax2.set_xlabel("Time since trigger (s)")
    
    fig.subplots_adjust(hspace=0)
    
    return fig
