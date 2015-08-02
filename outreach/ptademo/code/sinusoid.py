from __future__ import division
import numpy as np

def sinusoid(p,t):

    '''
    calculate sine function
    
    p - parameters (A, f, phi)
    t - discrete times
    '''

    y = p[0]*np.sin(2*np.pi*p[1]*t+p[2]) 

    return y
