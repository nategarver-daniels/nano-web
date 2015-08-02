from __future__ import division
import numpy as np

def removeoffset(ts, errors):
    
    '''
    detrend a time-series by removing constant offset

    b = best fit y-intercept
    '''
    
    N = len(ts[:,0])

    # b is given by weighted average of y's
    y = ts[:,1]
    sigma2 = errors[:,1]**2
    b = np.sum(y/sigma2) / np.sum(1/sigma2)

    # best fit line
    fit = b*np.ones(N)

    # detrended time-series
    dts = np.zeros([N,2])
    dts[:,0] = ts[:,0];
    dts[:,1] = ts[:,1] - fit
 
    return dts, b

