from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from correlate import *

def calmeasuredTOAs(ts, template, Tp):

    '''
    calculated measured TOAs and corresponding uncertainties by 
    correlating time-series with template
    '''

    DEBUG = 0

    # extract relevant time-domain quantities
    N = len(ts[:,0])
    deltaT = ts[1,0]-ts[0,0]
    deltaF = 1/(N*deltaT)
    fNyq = 1/(2*deltaT)
    
    # +/- indices for searching around peaks of correlation function
    Tcorr = 4e-4 # approx width of correlation peaks
    m = int(np.floor(Tcorr/deltaT))

    # fourier transform time-series and template data 
    ytilde = deltaT * np.fft.fft(ts[:,1])
    ptilde = deltaT * np.fft.fft(template[:,1])
    
    # correlate data (maxima give TOAs)
    # normalization implies that values of C at maxima are estimates of pulse amplitudes
    C = N * deltaF * np.fft.ifft(ytilde * np.conj(ptilde))
    norm = np.real(1/sum(deltaF * ptilde * np.conj(ptilde)))
    C = np.real(norm*C) # take real part to avoid imaginary round-off components

    # START DEBUG SECTION
    # plot correlation time series
    if DEBUG:
        plt.figure()
        plt.plot(ts[:,0], ts[:,1], '-b', ts[:,0], C, '-r')
        plt.xlabel('time (sec)')
        plt.legend(['time series', 'correlation'], 'best')
    # END DEBUG SECTION
        
    # find location of max correlation for reference pulse TOA
    ind0 = np.argmax(C)
    C0 = C[ind0]
    
    # use brent's method to find max
    print 'calculating reference TOA'

    if ind0 - m < 0: 
        indices = range(0, ind0+m+1)
    elif ind0 + m > len(C):
        indices = range(ind0-m, len(C))
    else:
        indices = range(ind0-m, ind0+m+1)

    brack = (ts[indices[0],0], ts[ind0,0], ts[indices[-1],0])
    t0 = opt.brent(correlate, (ts, template, -norm), brack, 1e-07, 0, 500)
    A0 = correlate(t0, ts, template, norm)

    # calculate expected number of pulses 
    N1 = int(np.floor(t0/Tp))
    N2 = int(np.floor((ts[-1,0]-t0)/Tp))
    Np = N1 + N2 + 1
    n0 = N1 + 1; # reference pulse number
    print 'reference TOA (n=', n0, ') has correlation=', A0

    # calculate expected indices of TOAs
    tmp = ind0 + np.round(np.linspace((Tp/deltaT)*(1-n0), (Tp/deltaT)*(Np-n0), Np));
    expected_ind = tmp.astype(int)

    # initialize variables
    tauhat = np.zeros(Np)
    Ahat = np.zeros(Np) 

    # loop to find measured TOAs and amplitudes
    for ii in range(0,Np):
    
        print 'calculating TOA', ii+1
        expected = expected_ind[ii]

        # search in a small region around expected arrival time
        if expected - m < 0: 
            indices = range(0,expected+m+1)
        elif expected + m > len(C):
            indices = range(expected-m, len(C))
        else:
            indices = range(expected-m, expected+m+1)
      
        # use brent's method to find max
        idx = indices[0] + np.argmax(C[indices])

        brack = (ts[indices[0],0], ts[idx,0], ts[indices[-1],0])

        try:
            tauhat[ii] = opt.brent(correlate, (ts, template, -norm), brack, 1e-07, 0, 500)
            badTOA = 0
        except:
            badTOA = 1
            print 'bad TOA'

        # set tauhat, Ahat to nan if brent's method can't find maximum
        if badTOA:
            tauhat[ii] = np.nan
            Ahat[ii] = np.nan
        else:
            Ahat[ii] = correlate(tauhat[ii], ts, template, norm)

        # START DEBUG SECTION
        if DEBUG:
            # plot data and correlation around max
            plt.figure()
            plt.plot(ts[indices,0], C[indices], '-*r', ts[indices,0], ts[indices,1], '-*b')
            plt.axhline(y = template[0,1], color='k') # starting value of pulse template
            TOAexp = t0 + (ii+1-n0)*Tp;
            plt.axvline(x = TOAexp, color='b')
            plt.axvline(x = tauhat[ii], color='r')
            plt.xlabel('time (sec)')
            plt.ylabel('correlation')
            plt.legend(['correlation', 'time series'], 'best')
            print tauhat[ii]-TOAexp
            #input('type any key to continue')
        # END DEBUG SECTION
    
    # error estimate for TOAs (based on correlation curve)
    # basically, we can determine the max of the correlation to +/- 0.5 deltaT;
    # multiply by 1/sqrt(Ahat(ii)) to increase error bar for small correlations

    ##Ahat_max = np.max(Ahat) # nan is max (which i don't want)
    Ahat_max = max(Ahat)

    error_tauhat = np.zeros(len(tauhat))
    for ii in range(0, len(tauhat)):
        error_tauhat[ii] = 0.5*deltaT/np.sqrt(Ahat[ii])
        
    # assign output variables (only TOAs and their uncertainties needed)
    measuredTOAs = tauhat
    uncertainties = error_tauhat

    return measuredTOAs, uncertainties, n0

