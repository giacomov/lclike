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

import ROOT

class FuncWrapper(ROOT.TPyMultiGenFunction):

    def __init__(self, function, dimensions):

        ROOT.TPyMultiGenFunction.__init__(self, self)
        self.function = function
        self.dimensions = int(dimensions)

    def NDim(self):
        return self.dimensions

    def DoEval(self, args):

        new_args = map(lambda i:args[i],range(self.dimensions))

        return self.function(*new_args)



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

    # FIx the n.a. if there is any
    with open("__temp_results.txt", "w+") as outf:

        with open(args.gtburst_results) as f:

            for line in f.readlines():

                outf.write(line.replace("n.a.", "0").replace("(fixed)",""))

    time_resolved_results = np.recfromtxt("__temp_results.txt", names=True)

    os.remove("__temp_results.txt")

    start = time_resolved_results['tstart']  # for iterating through bin directories
    stop = time_resolved_results['tstop']

    # Get start and stop of each bin and iterate through the results, loading the data

    with in_directory(args.directory):

        likelihood_objects = []

        for i, (bin_start, bin_stop) in enumerate(zip(start, stop)):

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

    # Fix all parameters for which the lower and the upper bound are equal

    new_initial_values = []
    new_boundaries = []

    for i, bounds in enumerate(boundaries):

        if bounds[0] == bounds[1] and bounds[0]!='none':

            logger.info("Fixing parameter %i (starting from 0)" % (i))

            # Parameter need to be fixed
            decay_function.parameters.values()[i].value = init_values[i]
            decay_function.parameters.values()[i].fix()

        else:

            new_initial_values.append(init_values[i])
            new_boundaries.append(boundaries[i])
            
            logger.info("parameter %i starts from %s with boundaries %s: " % (i, new_initial_values[-1], new_boundaries[-1]))

    decay_likelihood.setDecayFunction(decay_function)

    parameters_name = decay_function.getFreeParametersNames()

    logger.info("Setting up MINUIT fit...")
    
    functor = FuncWrapper(decay_likelihood.getLogLike, len(parameters_name))
    
    minimizer = ROOT.Math.Factory.CreateMinimizer("Minuit2", "Minimize")
    minimizer.Clear()
    minimizer.SetMaxFunctionCalls(1000)
    minimizer.SetTolerance(0.1)
    minimizer.SetPrintLevel(1)
    minimizer.SetErrorDef(0.5)
    
    minimizer.SetFunction(functor)
    
    for i, parameter_name in enumerate(parameters_name):
        
        if new_boundaries[i][0]=='none' and new_boundaries[i][1]=='none':
            
            minimizer.SetVariable(i, parameter_name, new_initial_values[i],
                                  abs(new_initial_values[i])/10.0)
        
        else:
        
            minimizer.SetLimitedVariable(i, parameter_name,
                                         new_initial_values[i],
                                         abs(new_initial_values[i])/10.0,
                                         new_boundaries[i][0],
                                         new_boundaries[i][1])
    
    minimizer.Minimize()
    
    #minimizer.Hesse()
    
    #minimizer.Minimize()
    
    best_fit_values = np.array(map(lambda x: x[0], zip(minimizer.X(), range(len(parameters_name)))))
    
#   ## Get MINOS errors
    minus_errors = []
    plus_errors = []
    
    for i, parameter_name in enumerate(parameters_name):
        
        eminus  = ROOT.Double ( 0 ) 
        eplus = ROOT.Double ( 0 ) 
        
        minimizer.GetMinosError(i, eminus, eplus)
    
    print("\n\n=================================================")
    print("Final results:")
    print("=================================================\n\n")
    
    for i, parameter_name in enumerate(parameters_name):
            
        print("%s: %s +%s" % (parameter_name, minus_errors[i], plus_errors[i]))
    
#    # Now prepare the arguments for Minuit
#    minuit_args = collections.OrderedDict()
#
#    # Forced_parameters tells minuit the name of the variable in the function, which
#    # are otherwise masqueraded by the use of *args
#    minuit_args['forced_parameters'] = parameters_name
#    
#    minuit_args['pedantic'] = True
#
#    # Use errordef = 0.5 which indicates that we are minimizing a -logLikelihood (and not chisq)
#    minuit_args['errordef'] = 0.5
#
#    for i, parameter_name in enumerate(parameters_name):
#
#        this_init = new_initial_values[i]
#        this_boundaries = new_boundaries[i]
#        this_delta = this_init
#
#        logger.info("Parameter %s: init_value = %s, boundaries = [%s,%s], delta = %s" % (parameter_name,
#                                                                                          this_init,
#                                                                                          this_boundaries[0],
#                                                                                          this_boundaries[1],
#                                                                                          this_delta))
#
#        minuit_args[parameter_name] = this_init
#        
#        if this_boundaries[0]=='none':
#            
#            this_boundaries[0] = None
#        
#        if this_boundaries[1]=='none':
#            
#            this_boundaries[1] = None
#        
#        minuit_args['limit_%s' % parameter_name] = this_boundaries
#        minuit_args['error_%s' % parameter_name] = this_delta
    
#    m = iminuit.Minuit(decay_likelihood.getLogLike, **minuit_args)
#    
#    #m.tol = 100
#    
#    logger.info("Performing MIGRAD minimization...")
#    
#    for i in range(10):
#    
#         res = m.migrad(resume=False, ncall=10000)
#    
#    logger.info("done")
#
#    logger.info("Performing MINOS ...")
#    
#    #m.tol = 1e-3
#    
#    _ = m.minos(maxcall=10000)
#
#    logger.info("done")
#
#    best_fit_values = map(lambda x: x['value'], res[1])

    filename = "%s_minuit_%s_results.txt" % (args.decay_function, args.triggername)

    logger.info("Saving results in %s..." % filename)

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
