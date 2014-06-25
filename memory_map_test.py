# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 23:32:22 2014

@author: leo
"""
import numpy as np
from ctypes import *

vector = np.arange(3*10*10, dtype=np.float64)
vector = vector.reshape((3,10,10))
flat = vector.flatten()

def pack(mem, array):
  flat = array.flatten()
  size = len(flat)
  for i in range(size):
    mem[i] = flat[i]

def unpack(mem, shape):
  flat = np.array([i for i in mem])
  return flat.reshape(shape)

mem = ( c_double * (10 * 10 * 3))()
pack(mem,vector)
vec2 = unpack(mem,(3,10,10))
	
for t in range(3):
	for y in range(10):	
		for x in range(10):	
			i = (t*10*10) + (y*10) + x
			assert mem[i]==vec2[t,y,x], "Fuck."