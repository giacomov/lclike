import collections
import math

import scipy.integrate
import scipy.optimize
import scipy.interpolate
import pyfits
import numpy

from Parameter import Parameter
import ggrb

INVALID = 1e6


class DecayFunction(object):
    def getParameters(self):
        return self.parameters

    def getFreeParameters(self):
        return filter(lambda x: x.isFree(), self.parameters.values())

    def getFreeParametersNames(self):
        return map(lambda x:x.name, self.getFreeParameters())

    def setParameters(self, *values):
        freeParameters = self.getFreeParameters()

        for i, v in enumerate(freeParameters):
            v.setValue(values[i])

    def getFlux(self, tmin, tmax):
        pass


class DecayBand(DecayFunction):
    def __init__(self):

        self.parameters = collections.OrderedDict()

        self.parameters['alpha'] = Parameter('alpha', 1, 0, 4, 0.1, fixed=False, nuisance=False, dataset=None)
        self.parameters['beta'] = Parameter('beta', -1.0, -4, 0, 0.1, fixed=False, nuisance=False, dataset=None)
        self.parameters['logT0'] = Parameter('logT0', 1, -3, 4, 0.1, fixed=False, nuisance=False, dataset=None,
                                             unit='s')
        self.parameters['logK'] = Parameter('logK', 1, -7, 24, 0.1, fixed=False, nuisance=False, dataset=None,
                                            normalization=False)

    def getDifferentialFlux(self, tt):

        # tt can be both a scalar or an array

        t = numpy.array(tt, ndmin=1, copy=False)

        alpha = self.parameters['alpha'].value
        beta = self.parameters['beta'].value
        logT0 = self.parameters['logT0'].value
        logK = self.parameters['logK'].value

        out = ggrb.ggrb(t, [alpha, beta, logT0, logK, 1e-5, 86400.0, 1])

        # This fixes nan(s) and inf values, converting them respectively to zeros and large numbers
        out = numpy.nan_to_num(out)
        if (out.shape[0] == 1):
            return max(0, out[0])
        else:
            return numpy.maximum(out, 0)

    def getFlux(self, tmin, tmax):

        # Integrate the differential flux between tmin and tmax

        tt = numpy.linspace(tmin, tmax, 51)
        ss = self.getDifferentialFlux(tt)

        res = scipy.integrate.simps(ss, tt)
        # res, err = scipy.integrate.quad( self.getDifferentialFlux, tmin, tmax, epsrel=1e-2,epsabs=0)

        return res / (tmax - tmin)

    def getPeakTimeAndFlux(self):

        a = self.parameters['alpha'].value

        # If a > 0, then the peak time is simply a*T0
        if a > 0:

            peak = a * pow(10, self.parameters['logT0'].value)

            return peak, self.getDifferentialFlux(peak)

        else:

            # General solution, slow

            fun = lambda x: self.getDifferentialFlux(x) * (-1)
            res = scipy.optimize.minimize_scalar(fun)

            return res['x'], self.getDifferentialFlux(res['x'])

    def getCharacteristicTime(self, fraction=0.367879):

        t0 = pow(10, self.parameters['logT0'].value)

        maxB = self.getDifferentialFlux(t0)

        biasedFlux = lambda t: self.getDifferentialFlux(t) - (maxB * fraction)

        # Interpolate to make this quicker
        interp_t = numpy.logspace(t0,6,1000)
        interp_y = biasedFlux(interp_t)

        interpolator = scipy.interpolate.InterpolatedUnivariateSpline(interp_t, interp_y, k=1)

        # Find root
        try:
            characteristicTime = scipy.optimize.brentq(interpolator, t0, 1e6, xtol=1e-3, rtol=1e-4)

        except:

            return -1

        else:

            return characteristicTime

    def getTsomething(self, what=90):

        frac = (100.0 - float(what)) / 2.0 / 100.0

        # Make an interpolated version of the model (way faster than integrating directly)

        interp_t = numpy.logspace(-5,numpy.log10(86400),1000)
        interp_y = self.getDifferentialFlux(interp_t)

        # Interpolate in the log space
        log_log_interpolator = scipy.interpolate.InterpolatedUnivariateSpline(numpy.log10(interp_t),
                                                                              numpy.log10(interp_y),
                                                                              k=1)



        # Make the integral distribution

        # Get the total integral
        f100, err = scipy.integrate.quad(self.getDifferentialFlux, 1e-5, 86400.0, epsrel=1e-2,epsabs=0)

        self.integralDistribution = lambda t: scipy.integrate.quad(self.getDifferentialFlux, 1e-5,
                                                                   t, epsrel=1e-2, epsabs=0)[0]

        self.int1 = lambda t: self.integralDistribution(t) - frac * f100

        try:

            t05 = scipy.optimize.brentq(self.int1, 1e-5, 86400.0, rtol=1e-2, maxiter=1000)

        except:

            t05 = -1

        self.int2 = lambda t: self.integralDistribution(t) - (1 - frac) * f100

        try:

            t95 = scipy.optimize.brentq(self.int2, t05, 86400.0, rtol=1e-2, maxiter=1000)

        except:

            t95 = -1

        return t95 - t05, t05, t95


class CrystalBall(DecayFunction):
    def __init__(self):

        self.parameters = collections.OrderedDict()

        self.parameters['mu'] = Parameter('mu',
                                          40.0,
                                          1e-2,
                                          200,
                                          5.0,
                                          normalization=False)

        self.parameters['sigma'] = Parameter('sigma',
                                             20.0,
                                             1e-2,
                                             40.0,
                                             5.0)
        self.parameters['t0'] = Parameter('t0',
                                          50.0,
                                          0.1,
                                          200,
                                          20.0,
                                          normalization=False)

        self.parameters['decayIndex'] = Parameter('decayIndex',
                                                  1.1,
                                                  0.3,
                                                  5.0,
                                                  0.1)
        self.parameters['N'] = Parameter('N',
                                         1e-2,
                                         1e-5,
                                         1,
                                         5e-3,
                                         normalization=True)

        # Shortcuts
        self.mu = self.parameters['mu'].getValue
        self.sigma = self.parameters['sigma'].getValue
        self.t0 = self.parameters['t0'].getValue
        self.decayIndex = self.parameters['decayIndex'].getValue
        self.N = self.parameters['N'].getValue

        self._defineIntegrals()

    def _getK(self):

        k = (math.exp(- pow(self.t0() - self.mu(), 2) / (2 * pow(self.sigma(), 2)))
             * pow(self.t0(), self.decayIndex())
             )

        return k

    def _powerLawIntegral(self, t):

        index = self.decayIndex()

        if (index != 1):

            return t * pow(t, -index) / (1 - index)

        else:

            return math.log(t)

    def _defineIntegrals(self):

        self.integral1stBranch_ = lambda t: (-1 * math.sqrt(math.pi / 2.0)
                                             * self.sigma()
                                             * math.erf((self.mu() - t) / (math.sqrt(2) * self.sigma()))
                                             )

        self.integral1stBranch = lambda t1, t2: self.N() * (self.integral1stBranch_(t2) -
                                                            self.integral1stBranch_(t1))

        self.integral2ndBranch_ = lambda t: self._getK() * self._powerLawIntegral(t)

        self.integral2ndBranch = lambda t1, t2: self.N() * (self.integral2ndBranch_(t2) -
                                                            self.integral2ndBranch_(t1))

    def getDifferentialFlux(self, t):

        # t = numpy.logspace(0,3,100)
        # y = map(cb.getDifferentialFlux, t)

        if (t <= self.t0()):

            return self.N() * math.exp(- pow(t - self.mu(), 2) / (2 * pow(self.sigma(), 2)))

        else:

            k = self._getK()

            return self.N() * k * pow(t, - self.decayIndex())

    def getFlux(self, tmin, tmax):

        if (tmax <= self.t0()):

            # All in first branch

            return self.integral1stBranch(tmin, tmax) / (tmax - tmin)

        elif (tmin > self.t0()):

            # All in second branch

            return self.integral2ndBranch(tmin, tmax) / (tmax - tmin)

        else:

            int1 = self.integral1stBranch(tmin, self.t0())
            int2 = self.integral2ndBranch(self.t0(), tmax)

            return (int1 + int2) / (tmax - tmin)


class CrystalBall2(CrystalBall):
    '''
    A version of the Crystal Ball function where mu = t0, i.e.,
    the decay starts at the peak time
    '''

    def __init__(self):
        self.parameters = collections.OrderedDict()

        self.parameters['mu'] = Parameter('mu',
                                          40.0,
                                          1e-2,
                                          200,
                                          5.0,
                                          normalization=False)
        self.parameters['sigma'] = Parameter('sigma',
                                             5.0,
                                             1e-2,
                                             200.0,
                                             1.0,
                                             normalization=False)

        self.parameters['decayIndex'] = Parameter('decayIndex',
                                                  1.1,
                                                  0.3,
                                                  5.0,
                                                  0.1)
        self.parameters['N'] = Parameter('N',
                                         1,
                                         1e-3,
                                         10,
                                         5e-3,
                                         normalization=False)

        # Shortcuts
        self.mu = self.parameters['mu'].getValue
        self.sigma = self.parameters['sigma'].getValue

        # In this version of the function, t0=mu
        self.t0 = self.mu

        self.decayIndex = self.parameters['decayIndex'].getValue
        self.N = self.parameters['N'].getValue

        self._defineIntegrals()


class DecayPowerlaw(DecayFunction):
    def __init__(self):

        self.parameters = collections.OrderedDict()

        self.parameters['decayIndex'] = Parameter('decayIndex',
                                                  1.1,
                                                  0,
                                                  10,
                                                  0.1)
        self.parameters['norm'] = Parameter("norm",
                                            1e-8,
                                            1e-32,
                                            100,
                                            1e-9)

        self.pivot = 1e2  # seconds

    def getFlux(self, tmin, tmax):

        index = self.parameters['decayIndex'].value

        if (index != 1):

            integral = lambda t: self.parameters['norm'].value * t * pow(t / self.pivot, -index) / (1 - index)

        else:

            integral = lambda t: self.parameters['norm'].value * self.pivot * math.log(t)

        return (integral(tmax) - integral(tmin)) / (tmax - tmin)


class DecayPowerlaw2(DecayFunction):
    '''
    Rise in flux is an increasing power law
    and is connected at the peak to a decreasing power law
    '''

    def __init__(self):

        self.parameters = collections.OrderedDict()

        self.parameters['decayIndex1'] = Parameter('decayIndex1',
                                                   1.1,
                                                   0,
                                                   10,
                                                   0.1)
        self.parameters['norm1'] = Parameter("norm1",
                                             1e-8,
                                             1e-32,
                                             100,
                                             1e-9)

        self.parameters['decayIndex2'] = Parameter('decayIndex2',
                                                   1.1,
                                                   0,
                                                   10,
                                                   0.1)
        self.parameters['norm2'] = Parameter("norm2",
                                             1e-8,
                                             1e-32,
                                             100,
                                             1e-9)

        self.pivot = 1e2  # seconds

    def getFlux(self, tmin, tmax):

        index = self.parameters['decayIndex2'].value

        if (index != 1):

            integral = lambda t: self.parameters['norm'].value * t * pow(t / self.pivot, -index) / (1 - index)

        else:

            integral = lambda t: self.parameters['norm'].value * self.pivot * math.log(t)

        return (integral(tmax) - integral(tmin)) / (tmax - tmin)


#############################################################
#############################################################

class likeObjectWrapper(object):
    def __init__(self, likeObject, srcName='GRB'):

        self.likeObject = likeObject

        # Check whether this likeObject crosses GTIs
        self.verifyGTIs()

        self.dt = map(lambda (t1, t2): t2 - t1, zip(self.tmin, self.tmax))

        self.srcName = srcName

        if (self.likeObject['GRB']['Spectrum'].func.genericName() != 'PowerLaw2'):
            raise RuntimeError("Only sources with PowerLaw2 spectrum are supported at the moment")

        # Force the energy range of the power law to be 100 MeV - 100 GeV

        self.likeObject[self.srcName]['Spectrum']['LowerLimit'] = 100.0
        self.likeObject[self.srcName]['Spectrum']['UpperLimit'] = 100000.0

        # Set Integral as fixed (probably useless, but oh well!)
        self.likeObject[self.srcName]['Spectrum'].params['Integral'].parameter.setAlwaysFixed(True)

        # Set the photon index to -2
        self.likeObject[self.srcName]['Spectrum'].params['Index'] = -2.0

        (self.fluxMinBound,
         self.fluxMaxBound) = self.likeObject[self.srcName]['Spectrum'].params['Integral'].parameter.getBounds()

    def isCrossingGTI(self):
        return self.crossesGTI

    def verifyGTIs(self):

        for eventfile in self.likeObject.observation.eventFiles:

            with pyfits.open(eventfile) as f:

                trigtime = f['EVENTS'].header.get("TRIGTIME")

                tstart = f['EVENTS'].header.get("TSTART") - trigtime
                tstop = f['EVENTS'].header.get("TSTOP") - trigtime

                gtis = f['GTI'].data
                gti_tstarts = gtis.field("START") - trigtime
                gti_tstops = gtis.field("STOP") - trigtime

            pass  # Close FITS file

            if (gti_tstarts.shape[0] > 1):
                self.crossesGTI = True

                self.tmin = sorted(list(gti_tstarts))

                self.tmax = sorted(list(gti_tstops))


                # raise RuntimeError("More than one GTIs for the event file %s" %(eventfile))

                # if( abs(tstart - gti_tstarts[0]) > 0.1 or
                #    abs(tstop - gti_tstops[0])   > 0.1 ):
                #  self.crossesGTI = True
                # raise RuntimeError("GTI and tstart-tstop in header do not match in %s" %(eventfile))

            else:
                self.crossesGTI = False
                self.tmin = [max(tstart, gti_tstarts[0])]
                self.tmax = [min(tstop, gti_tstops[0])]

        pass

    def setFlux(self, flux):

        # import pdb;pdb.set_trace()

        scale = self.likeObject[self.srcName]['Spectrum'].params['Integral'].parameter.getScale()

        # print("Scale is %s" %(scale))

        # print("Setting flux to %s for %s - %s" %(flux / scale , self.tmin, self.tmax))

        if flux <= self.fluxMinBound:

            # print("Flux is %s, minimum is %s" %(flux, self.fluxMinBound))

            value = self.fluxMinBound * 1.001

        elif (flux / scale >= self.fluxMaxBound * scale):

            # print("Flux is %s, maximum is %s" %(flux, self.fluxMaxBound))

            value = self.fluxMaxBound * 0.999

        else:

            value = flux

        self.likeObject[self.srcName]['Spectrum']['Integral'] = value / scale

    def getLogLike(self):

        self.likeObject.syncSrcParams()

        logl = self.likeObject.logLike.value()

        # print("Like is %s for %s - %s" %(logl, self.tmin, self.tmax))
        return logl


class DecayLikelihood(object):
    def __init__(self, *likeObjects):

        self.likeObjects = list(likeObjects)

    def setDecayFunction(self, function):

        self.decayFunction = function

    def getLogLike(self, *values):

        self.decayFunction.setParameters(*values)

        likeValues = []

        for likeObj in self.likeObjects:

            if (likeObj.isCrossingGTI()):

                # Compute the flux for the first GTI
                fluxes = []

                for t1, t2 in zip(likeObj.tmin, likeObj.tmax):
                    fluxes.append(self.decayFunction.getFlux(t1, t2))

                # Average them
                flux = numpy.average(fluxes)

            else:

                # Set the flux to the value foreseen by the decay function
                flux = self.decayFunction.getFlux(likeObj.tmin[0], likeObj.tmax[0])

            try:

                likeObj.setFlux(flux)

            except RuntimeError:

                return INVALID

            logl = likeObj.getLogLike()

            if (logl == INVALID):
                return INVALID

            likeValues.append(logl)

        return numpy.sum(likeValues) * (-1)
