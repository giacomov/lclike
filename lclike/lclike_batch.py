#!/usr/bin/env python

# ~/develop/lclike/lclike/lclike_batch.py --directory /home/giacomov/science/jarred/final_test/bn090510016_bins
# --triggername bn090510016
# --ft2file /home/giacomov/science/jarred/final_test/bn090510016_bins/bn090510016/gll_ft2_tr_bn090510016_v00.fit
# --gtburst_results /home/giacomov/science/jarred/final_test/bn090510016_bins/bn090510016_res.txt
# --decay_function band
# --initial_values '[2.5, -1.0, 1.0, -6]'
# --boundaries '[[1e-6, 30], [-5.0, -1e-6], [-3, 4], [-20, 20]]'
# --redshift 0.903
# --n_walkers 50
# --burn_in 100
# --steps 200

__author__ = 'giacomov'

import os
import contextlib
import logging
import ast
import collections
import argparse

import numpy as np

import iminuit
from UnbinnedAnalysis import *

from lclike import decayLikelihood
from lclike import plot_fit_results
from lclike import bayes_analysis

# Select a non-interactive matplotlib backend
import matplotlib
matplotlib.use("Agg")


@contextlib.contextmanager
def in_directory(directory):
    # Save current directory
    original_directory = os.getcwd()

    # Go to the requested directory

    os.chdir(directory)

    # Execute whatever we need
    yield

    # Go back to original directory
    os.chdir(original_directory)


# Set up logger
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# class dumb():
#    pass

mycmd = argparse.ArgumentParser()

mycmd.add_argument('--directory', help="Directory containing the file produced by gtburst", type=str, required=True)
mycmd.add_argument('--triggername', help='Name of the trigger', type=str, required=True)
mycmd.add_argument('--ft2file', help='The Ft2 file for the analysis', type=str, required=True)
mycmd.add_argument('--gtburst_results', help='Text file containing results from doTimeResolvedAnalysis.py', type=str,
                   required=True)
mycmd.add_argument('--decay_function', help="Function to model. (ex. crystalball2, band)", required=True)
mycmd.add_argument('--initial_values', help="Initial values for the parameters of the selected decay function. Ex:"
                                            " '[2.5, -1.0, 1, -6]'", required=True)
mycmd.add_argument('--boundaries', help="Boundaries for the parameters of the selected decay function. Ex:"
                                        " '[[1e-6, 30], [-5.0, -1e-6], [-3, 4], [-20, 20]]'", required=True)
mycmd.add_argument('--redshift', help='Redshift of this GRB (use 0 if unknown)', type=float, required=True)
mycmd.add_argument('--n_walkers', help='Number of walkers for the Bayesian Analysis', type=int, required=True)
mycmd.add_argument('--burn_in', help='Number of samples to throw away as part of the burn in', type=int, required=True)
mycmd.add_argument('--steps', help='Number of samples *per walker* to take from the posterior distribution', type=int,
                   required=True)
mycmd.add_argument('--minimum_time', help='Use only data after this time (optional)', type=float,
                   required=False, default=None)

# spacecraft data should be in same location as the analysis files############################################################################

if __name__ == "__main__":

    args = mycmd.parse_args()

    # args = dumb()

    # args.directory = '/home/giacomov/science/jarred/minuit/bn090510016_bins'
    # args.triggername = 'bn090510016'
    # args.ft2file = '/home/giacomov/science/jarred/minuit/bn090510016_bins/bn090510016/gll_ft2_tr_bn090510016_v00.fit'
    # args.gtburst_results = '/home/giacomov/science/jarred/minuit/bn090510016_bins/bn090510016_res.txt'
    # args.decay_function = 'band'
    # args.initial_values = '[2.5, -1.0, 1, -6]'
    # args.boundaries = '[[1e-6, 30], [-5.0, -1e-6], [-3, 4], [-20, 20]]'
    # args.redshift = 0.903
    # args.n_walkers = 50
    # args.burn_in = 20
    # args.steps = 50

    # Read in gtburst results

    time_resolved_results = np.recfromtxt(args.gtburst_results, names=True)

    # if args.minimum_time is not None:
    #
    #     idx = time_resolved_results['tstop'] > args.minimum_time
    #
    #     time_resolved_results = time_resolved_results[idx]

    start = time_resolved_results['tstart']  # for iterating through bin directories
    stop = time_resolved_results['tstop']

    # Get start and stop of each bin and iterate through the results, loading the data

    with in_directory(args.directory):

        likelihood_objects = []

        for i, (bin_start, bin_stop) in enumerate(zip(start, stop)):

            if args.minimum_time is not None:

                if bin_stop < args.minimum_time:

                    continue

            # Get the string corresponding to this time interval

            time_interval_string = "%s-%s" % (str(bin_start), str(bin_stop))

            logger.info("Loading interval %s (%s out of %s)" % (time_interval_string, i + 1, len(start)))

            # Now gather ft1 file, exposure map, livetime cube and XML model

            this_interval_dir = 'interval%s' % time_interval_string

            # We need to move into the directory because the XML file has relative paths

            with in_directory(this_interval_dir):

                ft1_file = 'gll_ft1_tr_%s_v00_filt.fit' % args.triggername

                exposure_map = 'gll_ft1_tr_%s_v00_filt_expomap.fit' % args.triggername

                livetime_cube = 'gll_ft1_tr_%s_v00_filt_ltcube.fit' % args.triggername

                xml_file = 'gll_ft1_tr_%s_v00_filt_likeRes.xml' % args.triggername

                # Check that they exist
                for this_file in [ft1_file, exposure_map, livetime_cube, xml_file]:

                    if not os.path.exists(this_file):

                        raise IOError("File %s does not exist in directory %s" % (this_file, this_interval_dir))

                # Create the likelihood object
                # NOTE: I use absolute paths here because we will need to reach these files even when we will not be
                # in the interval directory anymore

                this_obs = UnbinnedObs(os.path.abspath(ft1_file),
                                       os.path.abspath(args.ft2file),
                                       expMap=os.path.abspath(exposure_map),
                                       expCube=os.path.abspath(livetime_cube), irfs='CALDB')

                this_like = UnbinnedAnalysis(this_obs, xml_file, optimizer='Minuit')

            # Fix the galactic template to its best fit value to reduce the number of degrees of freedom

            this_like['GalacticTemplate']['Spectrum'].params['Value'].parameter.setFree(False)

            likelihood_objects.append(this_like)

            logger.info("done")

        logger.info("Loaded %s intervals" % len(likelihood_objects))

    # Now create the wrappers for the likelihood objects

    logger.info("Creating wrappers...")

    wrappers = []

    for likelihood_object in likelihood_objects:

        wrappers.append(decayLikelihood.likeObjectWrapper(likelihood_object))

    logger.info("done")

    logger.info("Setting up decay likelihood analysis...")

    # Now instance the DecayLikelihood object

    decay_likelihood = decayLikelihood.DecayLikelihood(*wrappers)

    # Default is Band

    decay_function = decayLikelihood.DecayBand()

    if args.decay_function == 'band':

        logger.info("Using DecayBand as decay function")

        decay_function = decayLikelihood.DecayBand()  # declaring instance of DecayLikelihood using Band

    elif args.decay_function == 'crystalball':

        logger.info("Using CrystalBall2 as decay function")

        decay_function = decayLikelihood.CrystalBall2()

    elif args.decay_function.lower() == 'willingale':

        logger.info("Using Willingale as decay function")

        decay_function = decayLikelihood.Willingale()

    elif args.decay_function.lower() == 'willingale2':

        logger.info("Using Willingale2 as decay function")

        decay_function = decayLikelihood.Willingale2()

    elif args.decay_function.lower() == 'powerlaw':

        logger.info("Using Powerlaw as decay function")

        decay_function = decayLikelihood.DecayPowerlaw()

    # Generate lists from the input strings by using literal_eval(), which is much
    # safer than eval() since it cannot evaluate anything which is not a variable
    # declaration

    init_values = list(ast.literal_eval(args.initial_values))
    boundaries = list(ast.literal_eval(args.boundaries))

    # Use 10% of the value as initial error (i.e. delta used by Minuit)

    errors = map(lambda x: abs(x / 10.0), init_values)

    # Fix all parameters for which the lower and the upper bound are equal

    new_initial_values = []
    new_boundaries = []
    new_errors = []

    for i, bounds in enumerate(boundaries):

        if bounds[0] == bounds[1]:

            logger.info("Fixing parameter %i (starting from 0)" % (i))

            # Parameter need to be fixed
            decay_function.parameters.values()[i].value = init_values[i]
            decay_function.parameters.values()[i].fix()

        else:

            new_initial_values.append(init_values[i])
            new_boundaries.append(boundaries[i])
            new_errors.append(errors[i])

    decay_likelihood.setDecayFunction(decay_function)

    parameters_name = decay_function.getFreeParametersNames()

    logger.info("Setting up MINUIT fit...")

    # Now prepare the arguments for Minuit
    minuit_args = collections.OrderedDict()

    # Forced_parameters tells minuit the name of the variable in the function, which
    # are otherwise masqueraded by the use of *args
    minuit_args['forced_parameters'] = parameters_name

    # Use errordef = 0.5 which indicates that we are minimizing a -logLikelihood (and not chisq)
    minuit_args['errordef'] = 0.5

    for i, parameter_name in enumerate(parameters_name):

        this_init = new_initial_values[i]
        this_boundaries = new_boundaries[i]
        this_delta = new_errors[i]

        logger.info("Parameter %s: init_value = %s, boundaries = [%s,%s], delta = %s" % (parameter_name,
                                                                                          this_init,
                                                                                          this_boundaries[0],
                                                                                          this_boundaries[1],
                                                                                          this_delta))

        minuit_args[parameter_name] = this_init
        minuit_args['limit_%s' % parameter_name] = this_boundaries
        minuit_args['error_%s' % parameter_name] = this_delta

    m = iminuit.Minuit(decay_likelihood.getLogLike, **minuit_args)

    logger.info("Performing MIGRAD minimization...")

    res = m.migrad()

    logger.info("done")

    logger.info("Performing HESSE ...")

    _ = m.hesse()

    logger.info("done")

    _ = m.migrad()

    logger.info("Performing MINOS ...")

    _ = m.minos()

    logger.info("done")

    # Activate optimization
    #decay_likelihood.activate_profiling()

    res = m.migrad()

    filename = "%s_minuit_%s_results.txt" % (args.decay_function, args.triggername)

    logger.info("Saving results in %s..." % filename)

    best_fit_values = map(lambda x: x['value'], res[1])

    with open(filename, "w+") as f:
        f.write("#")
        f.write(" ".join(parameters_name))
        f.write("\n")
        f.write(" ".join(map(lambda x: str(x), best_fit_values)))
        f.write("\n")

    logger.info("done")

    filename = "%s_minuit_%s_results.png" % (args.decay_function, args.triggername)

    logger.info("Plotting results in %s..." % filename)

    # Set the parameters to their best fit values
    for i, value in enumerate(best_fit_values):

        decay_function.parameters[parameters_name[i]].setValue(value)

    fig = plot_fit_results.plot_fit_results(time_resolved_results, args.redshift, args.triggername, decay_function)

    fig.savefig(filename)

    logger.info("done")

    if args.steps > 0:

        # Bayesian analysis

        logger.info("Setting up Bayesian analysis...")

        bayes = bayes_analysis.BayesianAnalysis(parameters_name, best_fit_values, decay_likelihood.getLogLike,
                                                new_boundaries)

        logger.info("done")

        logger.info(
            "Collecting %s samples with %s walkers and a burn in of %s..." % (args.steps, args.n_walkers, args.burn_in))

        bayes.sample(args.n_walkers, args.burn_in, args.steps)

        logger.info("done")

        filename = "%s_%s_samples.txt" % (args.decay_function, args.triggername)

        logger.info("Saving samples in file %s..." % filename)

        bayes.save_samples(filename)

        logger.info("done")

        logger.info("Saving traces plots...")

        figs = bayes.plot_traces()

        for parameter_name, fig in zip(parameters_name, figs):
            filename = "%s_%s_%s_trace.png" % (args.decay_function, args.triggername, parameter_name)

            fig.savefig(filename)

        filename = "%s_%s_traces.png" % (args.decay_function, args.triggername)

        logger.info("Saving corner plot in file %s..." % filename)

        fig = bayes.corner_plot()

        fig.savefig(filename)

    # Sample
