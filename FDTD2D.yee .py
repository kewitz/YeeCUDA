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

from scipy import misc

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
a.bound['Ez'][150:221,50:52] = 0

a.run(gauss,t=800)

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
	vmin,vmax = (vector.min(),vector.max())
	im = plt.imshow(vector[0,:,:], cmap='jet',vmin=vmin,vmax=vmax)
	fig.colorbar(im)
	ims = []
	for k in time:
		im = plt.imshow(vector[k,:,:], cmap='jet',vmin=vmin,vmax=vmax)
		ims.append([im])
	ani = animation.ArtistAnimation(fig, ims, interval=30, blit=True, repeat_delay=1000)
	plt.show()

#%%P
def snap2(vector):
	fig = plt.figure()
	im = plt.imshow(vector,cmap='jet')
	fig.colorbar(im)
	rect = plt.Rectangle((75, 75), 50, 50, facecolor="#ffffff", hatch="/")
	plt.gca().add_patch(rect)
	plt.xlabel('$x$')
	plt.ylabel('$y$')
	plt.show()

def snap1(vector):
	fig = plt.figure()
	plt.plot(vector)
	plt.axvspan(75, 125, facecolor='b', alpha=0.3)
	plt.xlabel('$x$')
	plt.ylabel('$Ez$')
	plt.show()
	
anim2D(a.Ez[0:400,:,:])
#%% Save Plot
#Writer = animation.writers['mencoder_file']
#writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)
#ani.save('img.mp4', writer=writer)
