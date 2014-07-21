/*
 * The MIT License (MIT)
 * Copyright (c) 2014 Leonardo Kewitz
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "snippets.cuh"

struct Yee{
	int lenX;
	int lenY;
	int lenT;
	double Z;
	double CEy;
	double CEx;
	double CH;
};

// Global index in Space domain.
__device__ int iS(int x, int y, int sx){
	int p = (y * sx) + x;
	return p;
}
// Global index in Space-Time domain.
__device__ int iST(int x, int y, int k, int sx, int sy){
	int p = x + (y*sx) + (k*sx*sy);
	return p;
}


__global__ void MagneticKernel(int lenX, int lenY, bool h, int k, double CEy, double CEx, double CH, double * Ez, double * Hx, double * Hy, long * boundEz, long * boundHx, long * boundHy) {
	// Reconhecimento espacial.
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= lenX || y >= lenY)
		return;

	int gw = lenX;
	int gh = lenY;
	int is = iS(x,y,gw);
	int ist = iST(x,y,k,gw,gh);

	long bHx = boundHx[is];
	long bHy = boundHy[is];

	double _hx = 0.0, _hy = 0.0;
	if (k > 0) {
		_hx = Hx[iST(x,y,k-1,gw,gh)];
		_hy = Hy[iST(x,y,k-1,gw,gh)];
	}
	double _ez = Ez[ist];
	// Calcula HX
	if (bHx == 1) {
		Hx[ist] = _hx - CH * (Ez[iST(x,y+1,k,gw,gh)] - _ez);
	}
	else if (bHx == 2) {
		if (y <= 1)
			Hx[ist] = Hx[iST(x,y+1,k-1,gw,gh)];
		else if (y >= lenY-2)
			Hx[ist] = Hx[iST(x,y-1,k-1,gw,gh)];
	}
	else
		Hx[ist] = 0.0;

	// Calcula HY
	if (bHy == 1) {
		Hy[ist] = _hy + CH * (Ez[iST(x+1,y,k,gw,gh)] - _ez);
	}
	else if (bHy == 2) {
		if (x <= 1)
			Hy[ist] = Hy[iST(x+1,y,k-1,gw,gh)];
		else if (x >= lenX-2)
			Hy[ist] = Hy[iST(x-1,y,k-1,gw,gh)];
	}
	else
		Hy[ist] = 0.0;

	return;
}

__global__ void ElectricKernel(int lenX, int lenY, bool h, int k, double CEy, double CEx, double CH, double * Ez, double * Hx, double * Hy, long * boundEz, long * boundHx, long * boundHy) {
	// Reconhecimento espacial.
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= lenX || y >= lenY)
		return;

	int gw = lenX;
	int gh = lenY;
	int is = iS(x,y,gw);
	int ist = iST(x,y,k,gw,gh);

	int bEz = boundEz[is];

	// Calcula Ez
	if (bEz == 1)
		Ez[iST(x,y,k+1,gw,gh)] = Ez[iST(x,y,k,gw,gh)] + (CEx * ( Hy[iST(x,y,k,gw,gh)] - Hy[iST(x-1,y,k,gw,gh)] )) - (CEy * (Hx[iST(x,y,k,gw,gh)] - Hx[iST(x,y-1,k,gw,gh)]));
	else if (bEz == 2) {
		if (y <= 1)
			Ez[iST(x,y,k+1,gw,gh)] = Ez[iST(x,y+1,k,gw,gh)];
		else if (x <= 1)
			Ez[iST(x,y,k+1,gw,gh)] = Ez[iST(x+1,y,k,gw,gh)];
		else if (x >= lenX-2)
			Ez[iST(x,y,k+1,gw,gh)] = Ez[iST(x-1,y,k,gw,gh)];
		else if (y >= lenY-2)
			Ez[iST(x,y,k+1,gw,gh)] = Ez[iST(x,y-1,k,gw,gh)];
	}


	return;
}


extern "C" int run(int lenX, int lenY, int lenT, double CEy, double CEx, double CH,  double * Ez, long * boundEz, long * boundHx, long * boundHy) {

	// Mallocs
	double * dEz, * dHx, * dHy;
	size_t s_ez = sizeof(double)*lenX*lenY*lenT;
	CudaSafeCall(cudaMalloc(&dEz, s_ez));
	CudaSafeCall(cudaMalloc(&dHx, s_ez));
	CudaSafeCall(cudaMalloc(&dHy, s_ez));

	long * dbEz, * dbHx, * dbHy;
	size_t bs = sizeof(long)*lenX*lenY;
	CudaSafeCall(cudaMalloc(&dbEz, bs));
	CudaSafeCall(cudaMalloc(&dbHx, bs));
	CudaSafeCall(cudaMalloc(&dbHy, bs));

	// Memcpy
	CudaSafeCall(cudaMemcpy(dEz, Ez, s_ez, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(dbEz, boundEz, bs, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(dbHx, boundHx, bs, cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(dbHy, boundHy, bs, cudaMemcpyHostToDevice));

	// Launch Settings
	const dim3 threads(BSIZE,BSIZE);
	const dim3 blocks(1 + lenX/threads.x, 1 + lenY/threads.y);
	for(int k = 0; k < lenT; k++) {
		MagneticKernel<<<blocks, threads>>>(lenX, lenY, true, k, CEy, CEx, CH, dEz, dHx, dHy, dbEz, dbHx, dbHy);
		cudaDeviceSynchronize();
		ElectricKernel<<<blocks, threads>>>(lenX, lenY, false, k, CEy, CEx, CH, dEz, dHx, dHy, dbEz, dbHx, dbHy);
		cudaDeviceSynchronize();
	}
	CudaCheckError();

	CudaSafeCall(cudaMemcpy(Ez, dEz, s_ez, cudaMemcpyDeviceToHost));
	CudaCheckError();

	cudaFree(dEz);
	cudaFree(dHx);
	cudaFree(dHy);
	cudaFree(dbEz);
	cudaFree(dbHx);
	cudaFree(dbHy);
	return 1;
}

extern "C" int checkCuda(){
  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
  return deviceCount;
}

void test() {
	double ez[500] = {};
	ez[1] = 2.0;
	double H[100] = {};
	int bound[100] = {};
	for (int i = 0; i < 100; ++i) {
		bound[i] = 1;
	}
}

int main(){
	int theresCuda = checkCuda();
	test();
	return 1;
}
