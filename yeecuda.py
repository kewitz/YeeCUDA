# -*- coding: utf-8 -*-
"""
The MIT License (MIT)
Copyright (c) 2014 Leonardo Kewitz

Created on Fri Jun 20 12:15:15 2014
"""
import time
import numpy as np
from ctypes import *
from scipy import misc

# Macros
indexed = lambda l, offset=0: zip(np.arange(len(l))+offset,l)

gpu = cdll.LoadLibrary('./Debug Shared/libYeeCUDA')  #Cuda
cpu = cdll.LoadLibrary('/home/leo/Code/OpenMP/libyeeOMP.so')  #CPU and OMP

# Constantes
f = np.float64  # Default format.
pi = f(np.pi)  # Double precision pi.
e = np.exp(f(1))  # Exponential as unit.

class YeeCuda:
  """ Classe de cálculo FDTD 2D. """
  def __init__(self, arg = False, verbose = True):
    self.vp = f(299792458.0)
    self.eps = f(8.854187817E-12)
    self.mu = f(pi * 4E-7)
    self.sigma = f(5E-15)
    self.verbose = verbose
    if type(arg) is str:
        image = misc.imread(arg, True)
        self.im = (image/255).round()
        self.sy, self.sx = list(self.im.shape)
    elif type(arg) is list:
        self.sx, self.sy = arg
        self.im = np.ones((self.sy,self.sx),dtype=np.int)
    else:
        raise Exception("Not a valid argument.")

  def setFreq(self, fop):
    """
    Ajusta os parâmetros da simulação de acordo com a frequência (Hz) de
    operação desejada.
    """
    # Define constantes:
    self.fop = fop
    self.lamb = self.vp/fop
    self.t0 = 1.0/fop
    self.tal = self.t0/(2*np.sqrt(2*np.log(2)))
    self.dx = self.lamb/8.0
    self.dy = self.lamb/8.0
    self.dtal = self.dx/2.0
    self.dt = self.dtal/self.vp

    # Domínios
    self.xd = np.arange(self.sx,dtype=np.float64)*self.dx
    self.yd = np.arange(self.sy,dtype=np.float64)*self.dy

    dbound = np.ones((self.sy,self.sx),dtype=np.int)
    dbound[0,:] = 0
    dbound[-1,:] = 0
    dbound[:,0] = 0
    dbound[:,-1] = 0

    self.bound = {
        'Ez': np.array(self.im, dtype=np.int),
        'Hx': dbound.copy(),
        'Hy': np.ones((self.sy,self.sx),dtype=np.int)
    }
    self.bound['Ez'][0,:] = 0
    self.bound['Ez'][-1,:] = 0
    self.bound['Ez'][:,0] = 0
    self.bound['Ez'][:,-1] = 0
    self.bound['Ez'][-1,1:-1] = 2
    self.bound['Ez'][0,1:-1] = 2
    

  def setLambda(self, lamb):
    """
    Ajusta os parâmetros da simulação de acordo com o comprimento de on-
    da (m) de operação desejado.
    """
    self.setFreq(self.vp/lamb)

  def makeDomains(self, iteractions, skip):
    self.td = np.arange(0,self.dt*iteractions,self.dt*skip,dtype=np.float64)
    self.Ez = np.zeros((len(self.td),self.sy,self.sx),dtype=np.float64)

  def run(self, fx, t=500, proc="GPU"):
    """ Executa o FDTD para os domínios configurados. """
    self.makeDomains(t, 1)
    # Constantes
    z = np.sqrt(self.mu/self.eps)
    CEy = z*self.dtal/self.dy
    CEx = z*self.dtal/self.dx
    CH = (1.0/z)*self.dtal/self.dx
    st,sx,sy = (len(self.td),self.sx,self.sy)
    if proc == "GPU":
        doit = gpu.run
    elif proc == "OMP":
        doit = cpu.runOMP
    elif proc == "CPU":
        doit = cpu.run
    else:
        print "'proc' arg must be 'GPU', 'CPU' or 'OMP'"

    if self.verbose: print "Recording input values..."
    fx(self)
    if self.verbose: print "Setting memory pointers..."
    cEz = np.ctypeslib.as_ctypes(self.Ez)
    boundEz = np.ctypeslib.as_ctypes(self.bound['Ez'])
    boundHx = np.ctypeslib.as_ctypes(self.bound['Hx'])
    boundHy = np.ctypeslib.as_ctypes(self.bound['Hy'])

    if self.verbose: print "Starting simulation..."
    start = time.time()
    doit(c_int(sx),c_int(sy),c_int(st),c_double(CEy),c_double(CEx),c_double(CH),byref(cEz),byref(boundEz),byref(boundHx),byref(boundHy))
    end = time.time()
    el = (end-start)
    if self.verbose: print "Simulation took %.2fs for %d nodes." % ((end-start), self.Ez.size)
    return el