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

#sim = YeeCuda('./caixa.png', verbose=False)
sim = YeeCuda([200,200], verbose=False)
cpu = []
omp = []
gpu = []
iters = 800
print "Starting benchmark with %d iterations." % iters
for s in range(5):
    sim.setFreq(2.4E9)
    sim.bound['Ez'][:,2] = 0
    sim.bound['Ez'][1,:] = 2
    sim.bound['Ez'][:,1] = 2
    sim.bound['Ez'][-1,:] = 2
    sim.bound['Ez'][:,-1] = 2
    t = sim.run(gauss,t=iters,proc="CPU")
    cpu.append(t)
    sim.setFreq(2.4E9)
    sim.bound['Ez'][:,2] = 0
    sim.bound['Ez'][1,:] = 2
    sim.bound['Ez'][:,1] = 2
    sim.bound['Ez'][-1,:] = 2
    sim.bound['Ez'][:,-1] = 2
    t = sim.run(gauss,t=iters,proc="OMP")
    omp.append(t)
    sim.setFreq(2.4E9)
    sim.bound['Ez'][:,2] = 0
    sim.bound['Ez'][1,:] = 2
    sim.bound['Ez'][:,1] = 2
    sim.bound['Ez'][-1,:] = 2
    sim.bound['Ez'][:,-1] = 2
    t = sim.run(gauss,t=iters,proc="GPU")
    gpu.append(t)

print "\tCPU\tGPU\tOMP"
print "Min\t%.4fs\t%.4fs\t%.4fs" %(np.min(cpu),np.min(gpu),np.min(omp))
print "Max\t%.4fs\t%.4fs\t%.4fs" %(np.max(cpu),np.max(gpu),np.max(omp))
print "Mean\t%.4fs\t%.4fs\t%.4fs" %(np.mean(cpu),np.mean(gpu),np.mean(omp))
print "OMP performance is %.2fx higher than CPU." %((np.mean(cpu)/np.mean(omp)))
print "GPU performance is %.2fx higher than OMP and %.2fx higher than CPU." %((np.mean(omp)/np.mean(gpu)),(np.mean(cpu)/np.mean(gpu)))

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
	
	ani = animation.FuncAnimation(fig, animate, time, interval=20)
	plt.show()
	return ani
	
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
	return ani

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
	
ani = anim2D(a.Ez)
#%% Save Plot
#Writer = animation.writers['mencoder_file']
#writer = Writer(fps=30, metadata=dict(artist='LKK'), bitrate=1800)
#ani.save('img.mp4', writer=writer)
