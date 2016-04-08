__author__ = 'giacomov'

# !/usr/bin/env python

# add |^| to the top line to run the script without needing 'python' to run it at cmd

# importing modules1
import numpy as np

# cant use 'show' inside the farm
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import gridspec

import os
import argparse

import decayLikelihood

import warnings

####################################################################

mycmd = argparse.ArgumentParser()  # this is a class
mycmd.add_argument('triggername', help="The name of the GRB in YYMMDDXXX format (ex. bn080916009)")
mycmd.add_argument('redshift', help="Redshift for object.")
mycmd.add_argument('function', help="Function to model. (ex. crystalball2, band)")

mycmd.add_argument('directory', help="Directory containing the file produced by gtburst")

if __name__ == "__main__":

    args = mycmd.parse_args()

    os.chdir(args.directory)

    ##############################################################################

    textfile = os.path.join(args.directory, '%s_res.txt' % (args.triggername))
    tbin = np.recfromtxt(textfile, names=True)

    textfile = os.path.join(args.directory, '%s_MCsamples_%s.txt' % (args.triggername, args.function))
    samples = np.recfromtxt(textfile, names=True)


    # function for returning 1 and 2 sigma errors from sample median
    def getErr(sampleArr):
        # compute sample percentiles for 1 and 2 sigma
        m, c, p = np.percentile(sampleArr, [16, 50, 84])
        # print("%.3f -%.3f +%.3f" %(c,m-c,p-c)) median, minus, plus
        m2, c2, p2 = np.percentile(sampleArr, [3, 50, 97])
        return m, c, p, m2, c2, p2


    # prepare for plotting and LOOP

    t = np.logspace(0, 4, 100)
    t = np.append(t, np.linspace(0, 1, 10))
    t.sort()
    t = np.unique(t)
    print('NUMBER OF times to iterate: %s' % (len(t)))

    x = decayLikelihood.DecayLikelihood()

    if args.function == 'crystalball2':

        crystal = decayLikelihood.CrystalBall2()  # declaring instance of DecayLikelihood using POWER LAW FIT
        x.setDecayFunction(crystal)

        # CrystalBall DiffFlux####################################################

        Peak = np.zeros(samples.shape[0])
        ePeak = np.zeros(samples.shape[0])
        tPeak = np.zeros(samples.shape[0])
        tePeak = np.zeros(samples.shape[0])

        print('ENTERING samples LOOP')

        # mu,sigma,decayIndex, and N
        for i, parameters in enumerate(samples):
            x.decayFunction.setParameters(*parameters)

            # NORMALIZATION IS THE FLUX AT THE PEAK
            pB = parameters[3]  # decay time is independent of scale # (y*.001) # scale =0.001, for all xml files
            fBe = pB / np.e

            # t = (fBe/N)**(-1/a) defined to be 1
            mu = parameters[0]
            tP = mu

            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    teP = mu + (fBe / parameters[3]) ** (
                    -1 / parameters[2])  # sometimes 'RuntimeWarning: overflow encountered in double_scalars'

                except Warning:
                    print('RuntimeWarning Raised! mu,sigma,decayIndex,and N:', parameters)

            teP = parameters[0] + (fBe / parameters[3]) ** (-1 / parameters[2])
            Peak[i] = pB
            ePeak[i] = fBe
            # redshift correcting t/(1+z)
            tPeak[i] = tP / (1 + float(args.redshift))  ################################
            tePeak[i] = teP / (1 + float(args.redshift))  ################################

    elif args.function == 'band':

        band = decayLikelihood.DecayBand()  # declaring instance of DecayLikelihood using POWER LAW FIT
        x.setDecayFunction(band)

        Peak = np.zeros(samples.shape[0])
        ePeak = np.zeros(samples.shape[0])  # fractional brightness used in calcuating char-time, but not needed otherwise
        tPeak = np.zeros(samples.shape[0])
        tePeak = np.zeros(samples.shape[0])  # characteristic time

        T05 = np.zeros(samples.shape[0])
        T90 = np.zeros(samples.shape[0])
        T95 = np.zeros(samples.shape[0])

        T25 = np.zeros(samples.shape[0])
        T50 = np.zeros(samples.shape[0])
        T75 = np.zeros(samples.shape[0])

        print('ENTERING samples LOOP')

        # mu,sigma,decayIndex, and N
        for i, parameters in enumerate(samples):
            x.decayFunction.setParameters(*parameters)

            tc = band.getCharacteristicTime()  # get the characteristic time.
            # T50/T90 TAKING TOO LONG (1/4)
            # t90, t05, t95 = band.getTsomething( 90 ) # if the argument is 90, returns the T90 as well as the T05 and the T95. If the argument is 50, returns the T50 as well as the T25 and T75, and so on.
            # t50, t25, t75 = band.getTsomething( 50 )

            tp, fp = band.getPeakTimeAndFlux()  # returns the time of the peak, as well as the peak flux

            tePeak[i] = tc / (1 + float(args.redshift))  ################################
            tPeak[i] = tp / (1 + float(args.redshift))
            Peak[i] = fp
            # T50/T90 TAKING TOO LONG (2/4)
            # T05[i] = t05/(1+float(args.redshift))
            # T90[i] = t90/(1+float(args.redshift))
            # T95[i] = t95/(1+float(args.redshift))
            # T50/T90 TAKING TOO LONG (3/4)
            # T25[i] = t25/(1+float(args.redshift))
            # T50[i] = t50/(1+float(args.redshift))
            # T75[i] = t75/(1+float(args.redshift))



    # Defining sigma bands
    print('ENTERING Percentile LOOP')
    upper = np.zeros(t.shape[0])
    lower = np.zeros(t.shape[0])
    upper2 = np.zeros(t.shape[0])
    lower2 = np.zeros(t.shape[0])
    meas = np.zeros(t.shape[0])
    fluxMatrix = np.zeros([samples.shape[0], t.shape[0]])

    for i, s in enumerate(samples):
        x.decayFunction.setParameters(*s)
        fluxes = map(x.decayFunction.getDifferentialFlux, t)
        fluxMatrix[i, :] = np.array(fluxes)

    for i, tt in enumerate(t):
        allFluxes = fluxMatrix[:, i]
        m, p = np.percentile(allFluxes, [16, 84])
        lower[i] = m
        upper[i] = p
        m2, p2 = np.percentile(allFluxes, [2.5, 97.5])
        lower2[i] = m2
        upper2[i] = p2

    wdir = '%s' % (args.directory)
    # save TXT files instead of .npy
    placeFile = os.path.join(wdir, "%s_tBrightness_%s" % (args.triggername, args.function))
    with open(placeFile, 'w+') as f:
        f.write("Peak tPeak ePeak tePeak\n")
        for i, s in enumerate(Peak):
            f.write("%s %s %s %s\n" % (Peak[i], tPeak[i], ePeak[i], tePeak[i]))
    # CALCULATING T50/T90 TAKES TOO LONG
    # T50/T90 TAKING TOO LONG (4/4)
    # if args.function == 'band':
    #  #compute percentiles for 1 sigma
    #  m90,c90,p90 = np.percentile(T90,[16,50,84])
    #  m50,c50,p50 = np.percentile(T50,[16,50,84])
    #  #compute percentiles for 1 and 2 sigma
    #  #90m,90c,90p,90m2,90c2,90p2 = getErr(T90)
    #  #50m,50c,50p,50m2,50c2,50p2 = getErr(T50)
    #  #print("%.3f -%.3f +%.3f" %(c,m-c,p-c)) median, minus, plus
    #
    #  placeFile=os.path.join(wdir,"%s_t90_t50_%s" % (args.triggername, args.function) )
    #  with open(placeFile,'w+') as f:
    #    f.write("t90 90minus 90plus t50 50minus 50plus\n")
    #    for i,s in enumerate(T90):
    #      f.write("%s %s %s %s %s %s\n" % (m90,m90-c90,p90-c90,c50,m50-c50,p50-c50)) #c,m-c,p-c
    #
    #  placeFile=os.path.join(wdir,"%s_samplesT90_%s" % (args.triggername, args.function) )
    #  with open(placeFile,'w+') as f:
    #    f.write("t90 t05 t95\n")
    #    for i,s in enumerate(T90):
    #      f.write("%s %s %s\n" % (T90[i],T05[i],T95[i]))

    #  placeFile=os.path.join(wdir,"%s_samplesT50_%s" % (args.triggername, args.function) )
    #  with open(placeFile,'w+') as f:
    #    f.write("t50 t25 t25\n")
    #    for i,s in enumerate(T50):
    #      f.write("%s %s %s\n" % (T50[i],T25[i],T75[i]))

    # compute char-time percentiles for 1 and 2 sigma
    m, c, p, m2, c2, p2 = getErr(tePeak)

    # saves txt file
    wkdir = '%s' % (args.directory)

    fileDir = os.path.join(wkdir, '%s_timeRes_%s' % (args.triggername, args.function))
    with open(fileDir, 'w+') as f:
        f.write('%s %s %s\n' % ('median', 'minus', 'plus'))
        f.write('%s %s %s\n' % (c, m - c, p - c))

    # PLOTTING BINS AND SIGMA BAND
    print("PLOTTING...")
    fig = plt.figure()

    # median is your "x"
    # Y is your "y"
    # DY is the array containing the errors
    # DY==0 filters only the zero error

    data = tbin
    # redshift correction /(1+args.redshif)
    median = (data["tstart"] + data["tstop"]) / 2 / (1 + float(args.redshift))
    start = data['tstart'] / (1 + float(args.redshift))  ##
    stop = data['tstop'] / (1 + float(args.redshift))  ##

    y = data["photonFlux"]
    Dy = data["photonFluxError"]

    try:
        y = np.core.defchararray.replace(y, "<", "", count=None)  # runs through array and removes strings
    except:
        print('No Upper-Limits Found in %s.' % (args.triggername))

    try:
        Dy = np.core.defchararray.replace(Dy, "n.a.", "0",
                                          count=None)  ## 0 error is nonphysical, and will be checked for in plotting
    except:
        print('No 0-Error Found in %s.' % (args.triggername))

    bar = 0.5
    color = "blue"

    Y = np.empty(0, dtype=float)  # makes empty 1-D array for float values

    for i in y:
        Y = np.append(Y, float(i))

    DY = np.empty(0, dtype=float)

    for i in Dy:
        DY = np.append(DY, float(i))
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

    plt.suptitle('%s photonFlux per Time' % (args.triggername))
    plt.xlabel('Rest Frame Time(s)')
    plt.ylabel('Photon Flux')
    plt.xscale('symlog')
    plt.yscale('log')
    plt.grid(True)

    if args.function == 'crystalball2':
        SCALE = 0.001
    elif args.function == 'band':
        SCALE = 1.0  # 0.1 # shouldn't need a scale anymore for Band function

    ylo = 1e-7  # min(lower2*SCALE)*1e-1 # CANT GET THIS TO WORK YET DYNAMICALLY
    yup = max(upper2 * SCALE) * 10
    plt.ylim([ylo, yup])

    # correcting for redshift t/(1+args.redshift)
    plt.fill_between(t / (1 + float(args.redshift)), lower * SCALE, upper * SCALE, alpha=0.5, color='blue')
    plt.fill_between(t / (1 + float(args.redshift)), lower2 * SCALE, upper2 * SCALE, alpha=0.3, color='green')
    # y = map(x.decayFunction.getDifferentialFlux, t) # maps infinitesimal values of flux at time t to y

    # raw_input("Press ENTER")
    # PowerLaw
    # plt.plot(t,,'o')

    # saves plots
    wdir = '%s' % (args.directory)

    imsave = os.path.join(wdir, '%s_objFit_%s' % (args.triggername, args.function))

    plt.savefig(imsave + '.png')

    # histograms of 1/e and save
    print("Making histograms")
    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

    bins = np.linspace(min(tePeak), np.max(tePeak), 100)

    ax0 = plt.subplot(gs[0])
    ax0.hist(tePeak, bins, normed=True)
    plt.title('1/e (min to medx2)')
    plt.xlabel('1/e time (s)')
    plt.xlim([min(tePeak), np.median(tePeak) * 2])

    ax1 = plt.subplot(gs[1])

    ax1.hist(tePeak, bins, normed=True)
    plt.title('1/e (min to max)')
    plt.xlabel('time (s)')

    plt.tight_layout()

    imsave = os.path.join(wdir, '%s_hist_%s' % (args.triggername, args.function))

    plt.savefig(imsave + '.png')
    print("Finished Potting/Saving!")
