__author__ = 'giacomov'

import numpy as np
import emcee
import matplotlib.pyplot as plt
import corner


class BayesianAnalysis(object):
    def __init__(self, parameter_names, best_fit_values, loglike, boundaries):

        self.parameter_names = parameter_names

        self.loglike = loglike
        self.boundaries = boundaries

        self.n_dim = len(boundaries)
        self.best_fit_values = best_fit_values

        self._chain = None
        self._samples = None

    @property
    def samples(self):

        return self._samples

    def lnprob(self, theta):  # , x, y, yerr):

        lp = self.logprior(theta)

        if not np.isfinite(lp):
            return -np.inf

        return lp + (-1) * self.loglike(*theta)

    @staticmethod
    def get_init_positions(best_fit_values, boundaries, n_walkers):
        """
        Returns init positions for the walkers, randomized around the best fit values
        """

        init_positions = np.zeros([n_walkers, len(best_fit_values)])

        for i, boundary in enumerate(boundaries):

            v = best_fit_values[i]

            this_low_bound = boundary[0]
            this_hi_bound = boundary[1]

            if boundary[0] < v < boundary[1]:

                this_low_bound = max(boundary[0], v - 0.1 * abs(v))

                this_hi_bound = min(boundary[1], v + 0.1 * abs(v))

            elif v <= boundary[0]:

                # We are at the lower limit for this parameter

                this_low_bound = boundary[0] + 0.1 * abs(boundary[0])

                this_hi_bound = boundary[0] + 0.2 * abs(boundary[0])

            elif v >= boundary[1]:

                # We are at the lower limit for this parameter

                this_low_bound = boundary[1] - 0.2 * abs(boundary[1])

                this_hi_bound = boundary[1] - 0.1 * abs(boundary[1])

            init_positions[:, i] = np.random.uniform(this_low_bound, this_hi_bound, n_walkers)

            # print("par. value %s, min %s, max %s" % (v, init_positions[:, i].min(),
            #                                         init_positions[:, i].max()))

        # Fix with respect to the other constraints
        for i in range(n_walkers):

            mu = init_positions[i, 0]

            if init_positions[i, 1] >= mu:
                print("Fixing sample %i" % (i))

                init_positions[i, 1] = np.random.uniform(max(boundaries[0][0], mu - 0.1 * mu), mu)

        return init_positions

    def logprior(self, theta):

        assert len(theta) == self.n_dim, "Dimension of parameters' vector does not match with the number of parameters"

        for this_value, boundary in zip(theta, self.boundaries):

            if this_value < boundary[0] or this_value > boundary[1]:
                return -np.inf

        return 0

    def sample(self, n_walkers, burn_in, steps):

        # Get init position
        p0 = self.get_init_positions(self.best_fit_values, self.boundaries, n_walkers)

        sampler = emcee.EnsembleSampler(n_walkers, self.n_dim, self.lnprob)

        pos, prob, state = sampler.run_mcmc(p0, burn_in)

        sampler.reset()

        sampler.run_mcmc(pos, steps)

        self._chain = sampler.chain
        self._samples = sampler.chain[:, :, :].reshape((-1, self.n_dim))

        return self._chain

    def plot_traces(self):

        labels = self.parameter_names

        figs = []

        for j in range(self.n_dim):

            fig = plt.figure()  # figsize=(15,15.0/3.3))

            for i in range(self._chain.shape[0]):
                plt.plot(self._chain[i][:, j], ',', c='black')

                plt.title(labels[j])

            figs.append(fig)

        return figs

    def corner_plot(self):

        fig = corner.corner(self.samples, labels=self.parameter_names,
                            truths=None)

        return fig

    def save_samples(self, filename):

        with open(filename, 'w+') as f:

            f.write("%s\n" % " ".join(self.parameter_names))

            for i,s in enumerate(self.samples):

                for p in s:

                    f.write(" %s" % p)

                f.write('\n')

# lnlike = decay_likelihood.getLogLike
#
# ndim, nwalkers = 4, 100
#
#
#
#
#
# alpha, beta, logT0, logK = map(lambda x: x['value'], res[1])
#
# print(alpha, beta, logT0, logK)
#
# init_result = np.array([alpha, beta, logT0, logK])
#
# pos = getInitPositions(init_result, boundaries, nwalkers)
#
# # Ensure we have a legal start
#
# priorsval = map(lnprior, pos)
#
# print priorsval
#
# for i, pp in enumerate(priorsval):
#     # print pp
#     if not np.isfinite(pp):
#         raise RuntimeError("Illegal start")
#
#
# # In[40]:
#
# # x.setInnerFit( True )
#
# import emcee
#
# sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
#
# steps = 50
# burn_in = 20
# print("running sampler for %s steps" % (steps))
# sampler.run_mcmc(pos, steps)
#
#
# # In[42]:
#
# # plot trace of walkers
# labels = parameters_name
# wdir = '%s' % (args.directory)
# for j in range(ndim):
#
#     fig = plt.figure()  # figsize=(15,15.0/3.3))
#     for i in range(sampler.chain.shape[0]):
#         plt.plot(sampler.chain[i][:, j], '.', c='black')
#         plt.title(labels[j])
