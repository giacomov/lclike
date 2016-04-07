from scipy.special import gammaincc, gamma

from math import exp, pow, log

import numpy

def ggrb_int_cpl( a, Ec, Emin, Emax):

    i1 = gammaincc(2 + a, Emin / Ec) * gamma(2+a)
    i2 = gammaincc(2 + a, Emax / Ec) * gamma(2+a)
    
    return -Ec * Ec * (i2 - i1)

def ggrb_int_pl(a, b, Ec, Emin, Emax):
    
    pre = pow(a-b, a-b) * exp(b-a) / pow(Ec, b)
    
    if b != -2:
        
        return pre / (2+b) * (pow(Emax, 2+b) - pow(Emin, 2+b))
    
    else:
        
        return pre * log(Emax / Emin)

def ggrb( energy, parameter):
    
    a = parameter[0]
    b = parameter[1]
    log_Ep = parameter[2]
    log_Flux = parameter[3]
    Emin = parameter[4]
    Emax = parameter[5]
    opt = parameter[6]
    
    #Cutoff energy
    
    if a == -2:
        
        Ec = pow(10, log_Ep) / 0.0001 #TRICK: avoid a=-2
    
    else:
        
        Ec = pow(10, log_Ep) / (2 + a)
    
    #Split energy
    
    Esplit = (a-b) * Ec
    
    #Evaluate model integrated flux and normalization
    
    if opt==0:
        
        #Cutoff power law
        
        intflux = ggrb_int_cpl(a, Ec, Emin, Emax)
    
    else:
        
        #Band model
        
        if Emin <= Esplit and Esplit <= Emax:
            
            intflux = ( ggrb_int_cpl(a,    Ec, Emin,   Esplit) +
                        ggrb_int_pl (a, b, Ec, Esplit, Emax) )
        
        else:
            
            if Esplit < Emin:
                
                intflux = ggrb_int_pl(a, b, Ec, Emin, Emax)
            
            else:
                
                raise RuntimeError("Esplit > emax!")
    
    erg2keV = 6.24151e8
    
    norm = pow(10, log_Flux) * erg2keV / intflux
    
    if opt==0:
    
        #Cutoff power law
        
        flux = norm * numpy.power(energy / Ec, a) * numpy.exp( - E / Ec)
    
    else:
        
        idx = (energy < Esplit)
        
        flux = numpy.zeros_like( energy )
        
        flux[idx] = ( norm * numpy.power(energy[idx] / Ec, a) * 
                      numpy.exp(-energy[idx] / Ec) )
        
        nidx = ~idx
        
        flux[nidx] = ( norm * pow(a-b, a-b) * exp(b-a) * 
                       numpy.power(energy[nidx] / Ec, b) )
    
    return flux
