# -*- coding: utf-8 -*-
"""
The MIT License (MIT)
Copyright (c) 2014 Leonardo Kewitz

Simulação do erro de dispersão do FDTD quando não considerado o passo mágico em
um impulso gaussiano modulado em seno.

Created on Wed May 28 11:11:30 2014
@author: leo
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from yeecuda import YeeCuda

# Macros
indexed = lambda l, offset=0: zip(np.arange(len(l))+offset,l)

def gauss(fdtd):
	width = (2*np.power(fdtd.tal,2))
	func = lambda t: np.exp(-np.power(t-2*fdtd.t0,2) / width)
	for k,t in indexed(fdtd.td):
		fdtd.Ez[k,:,2] = func(t)

a = YeeCuda()
a.setFreq(2.4E9)

a.bound['Ez'][:,2] = 2

a.bound['Ez'][75:126,75:126] = 0
a.bound['Hy'][75:126,75] = 0
a.bound['Hy'][75:126,125] = 0
a.bound['Hx'][75,75:126] = 0
a.bound['Hx'][125,75:126] = 0

a.run(gauss,t=1000)

#%%Plot
def anim1D(vector, time=None):
	time = np.arange(len(vector[:,0])) if time is None else time
	fig, ax = plt.subplots()
	line, = ax.plot(vector[0,:])
	plt.ylim(vector.min(),vector.max())
	plt.grid()

	def animate(s):
		line.set_ydata(vector[s,:])
		return line,
	
	animate = animation.FuncAnimation(fig, animate, time, interval=20)
	plt.show()
	
def anim2D(vector, time=None):
	time = np.arange(len(vector[:,0,0])) if time is None else time
	fig = plt.figure()
	ims = []
	vmin,vmax = (vector.min(),vector.max())
	for k in time:
		im = plt.imshow(vector[k,:,:],vmin=vmin, vmax=vmax)
		ims.append([im])
	ani = animation.ArtistAnimation(fig, ims, interval=30, blit=True, repeat_delay=1000)
	plt.show()

anim2D(a.Ez, np.arange(0,len(a.td),3))
print a.bound['Ez'], 'Ez'
print a.bound['Hx'], 'Hx'
print a.bound['Hy'], 'Hy'
#%% Save Plot
#Writer = animation.writers['mencoder_file']
#writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)
#ani.save('img.mp4', writer=writer)
