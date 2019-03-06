#coding:utf-8

# A class of wavelet (Daubechies) transform and inverse transform 
#
# decompose to elements by Wavelet transform
# and, compose from selected elements except mainly noise elements by inverse Wavelet transform 
#
# example: python3 wavelet1.py

# Check version
#  Python 3.6.4 on win32 (Windows 10)
#  numpy 1.14.0 
#  scipy 1.0.0
#  matplotlib  2.1.1


import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


class Class_Wavelet1(object):
    def __init__(self, N_DIMENSION=3):
        # initalize
        self.N_DIMENSION= N_DIMENSION  # order of Wavelet Daubechies
        self.p0= signal.wavelets.daub(self.N_DIMENSION) # get coefficient
        self.q0= np.copy(self.p0[::-1])
        for i in range(len(self.p0)):
            self.q0[i]=   -1.0 * self.q0[i] * (np.power(-1.0, i+1))
        
    def dwt1d(self, s0, g, h):
        """
        discrete wavelet transform (1-dimension)
        
        argument:
            s0: scaling coefficient level-0,  len(s0) must be even number.
            g: decomposition G(k)
            h: decomposition H(k)
        return:
            s1: scaling coefficient level-1
        """
        s_length_half= int(len(s0)/2)
        s1_LPF = np.zeros(s_length_half)
        s1_HPF = np.zeros(s_length_half)
        
        for n in range (s_length_half):
            for k in range (len(g)):
                i= np.mod((2*n+k) , len(s0))
                s1_LPF[n] = s1_LPF[n] + g[k] * s0[i]
                s1_HPF[n] = s1_HPF[n] + h[k] * s0[i]
        
        s1= np.concatenate( [s1_LPF, s1_HPF ] )
        
        return s1

    def idwt1d(self, s1, p, q):
        """
        inverse discrete wavelet transform (1-dimension)
        
        argument:
            s1: scaling coefficient level-1,  len(s1) must be even number.
            p: composition P(k)
            q: composition Q(k)
        return:
            s0: scaling coefficient level-0
        """
        # check
        if len(s1) < len(p):
            print ('error: len(s1) < len(p)')
        
        s_length_half = int(len(s1)/2)
        s0_LPF = np.zeros(len(s1))
        s0_HPF = np.zeros(len(s1))
        s0 = np.zeros(len(s1))
        
        for n in range (s_length_half):
            s0_LPF[2*n] = s1[n]
            s0_HPF[2*n] = s1[n + s_length_half]
            
        for n in range (len(s1)):
            for k in range (len(p)):
                i = np.mod( (n-k+len(s1)), len(s1)  )
                s0[n] = s0[n] + p[k] * s0_LPF[i] + q[k] * s0_HPF[i]
        
        return s0
        
    def trans_itrans_level8(self, s0, filter=None, show=False):
        # decompose to elements by Wavelet transform
        # and, compose from selected elements except mainly noise elements by inverse Wavelet transform 
        # This is level upto 8
        
        # check
        if len(s0) < ( np.power(2, 8-1) * len(self.p0)):
            print ('error: input length is not enough.')
        
        s1 = self.dwt1d(s0, self.p0, self.q0)                      # 1/1
        s2 = self.dwt1d(s1[0:int((len(s1)/2))], self.p0, self.q0)  # 1/2
        s3 = self.dwt1d(s2[0:int((len(s2)/2))], self.p0, self.q0)  # 1/4
        s4 = self.dwt1d(s3[0:int((len(s3)/2))], self.p0, self.q0)  # 1/8
        s5 = self.dwt1d(s4[0:int((len(s4)/2))], self.p0, self.q0)  # 1/16
        s6 = self.dwt1d(s5[0:int((len(s5)/2))], self.p0, self.q0)  # 1/32
        s7 = self.dwt1d(s6[0:int((len(s6)/2))], self.p0, self.q0)  # 1/64
        s8 = self.dwt1d(s7[0:int((len(s7)/2))], self.p0, self.q0)  # 1/128
        
        # select switch (filter):  if value is 0.0: then doesn't use the element
        # element: s1,  s2,  s3,  s4,  s5,  s6,  s7,  s8
        if filter is None:
            flt=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        else:
            flt=filter
        
        is7 = self.idwt1d(s8 * flt[7] , self.p0, self.q0)
        iss7 = np.concatenate([is7,s7[len(is7):] * flt[6] ])
        is6 = self.idwt1d(iss7, self.p0, self.q0)
        iss6 = np.concatenate([is6,s6[len(is6):] * flt[5] ])
        is5 = self.idwt1d(iss6, self.p0, self.q0)
        iss5 = np.concatenate([is5,s5[len(is5):] * flt[4] ])
        is4 = self.idwt1d(iss5, self.p0, self.q0)
        iss4 = np.concatenate([is4,s4[len(is4):] * flt[3] ])
        is3 = self.idwt1d(iss4, self.p0, self.q0)
        iss3 = np.concatenate([is3,s3[len(is3):] * flt[2] ])
        is2 = self.idwt1d(iss3, self.p0, self.q0)
        iss2 = np.concatenate([is2,s2[len(is2):] * flt[1] ])
        is1 = self.idwt1d(iss2, self.p0, self.q0)
        iss1 = np.concatenate([is1,s1[len(is1):] * flt[0] ])
        is0 = self.idwt1d(iss1, self.p0, self.q0)
        
        if show:
            self.trans_itrans_show(s0,is0, s1, s2, s3, s4, s5, s6, s7, s8)
        
        return is0  # output of transform and inverse transform with select switch (filter)
        
    def trans_itrans_show(self,s0, is0, s1, s2, s3, s4, s5, s6, s7, s8):
        # draw 
        fig = plt.figure()
        plt.subplot(2,1,1)
        plt.xlabel('step')
        plt.ylabel('level')
        plt.title( 'blue: original  red: composition' )
        plt.plot( np.arange(len(s0))  ,  s0,  color='b')
        plt.plot( np.arange(len(is0)) , is0, color='r')
        plt.grid()
        
        plt.subplot(2,1,2)
        plt.xlabel('step')
        plt.ylabel('level')
        plt.title( 'elements: s1, s2, s3, s4, s5, s6, s7, s8' )
        plt.plot( s1,'k' )
        plt.plot( s2,'b' )
        plt.plot( s3,'g' )
        plt.plot( s4,'r' )
        plt.plot( s5,'c' )
        plt.plot( s6,'m' )
        plt.plot( s7,'y' )
        plt.plot( s8,'k' )
        plt.grid()
        
        fig.tight_layout()
        plt.show()
        

if __name__ == '__main__':
    
    from scipy.io.wavfile import read as wavread
    
    # instance
    w=Class_Wavelet1() 
    
    # load wav sample
    path='sample1.wav'
    sr, x = wavread(path)
    
    # (1) show back to original waveform by transform and inverse transform
    # select switch (filter):  if value is 0.0: then doesn't use the element
    #    s1,  s2,  s3,  s4,  s5,  s6,  s7,  s8
    flt=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] # use all elements
    lng0=2048 # set length of wavelet transform
    y= w.trans_itrans_level8( x[0:lng0], filter=flt, show=True )
    
    # (2) show comparison with composition from selected elements only
    # select switch (filter):  if value is 0.0: then doesn't use the element
    #    s1,  s2,  s3,  s4,  s5,  s6,  s7,  s8
    flt=[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]  # only s6, s7, and s8 are used
    lng0=2048
    y= w.trans_itrans_level8( x[0:lng0], filter=flt, show=True )
    
    
    # load another wav sample
    path='sample2.wav'
    sr, x = wavread(path)
    
    # (1) show back to original waveform by transform and inverse transform
    # select switch (filter):  if value is 0.0: then doesn't use the element
    #    s1,  s2,  s3,  s4,  s5,  s6,  s7,  s8
    flt=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] # use all elements
    lng0=2048
    y= w.trans_itrans_level8( x[0:lng0], filter=flt, show=True )
    
    # (2) show comparison with composition from selected elements only
    # select switch (filter):  if value is 0.0: then doesn't use the element
    #    s1,  s2,  s3,  s4,  s5,  s6,  s7,  s8
    flt=[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]  # only s6, s7, and s8 are used
    lng0=2048
    y= w.trans_itrans_level8( x[0:lng0], filter=flt, show=True )
    
    