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
__device__ int iS(int x, int y, int grid_width){
	int p = (y * grid_width) + x;
	return p;
}
// Global index in Space-Time domain.
__device__ int iST(int x, int y, int k, int sx, int sy){
	int p = x + (y*sx) + (k*sx*sy);
	return p;
}

__device__ int calcE(int x, int y, int lenX, int lenY) {
	int r = 0;
	if (x > 0 && y > 0 && x < lenX - 1 && y < lenY -1)
		r = 1;
	return r;
}
__device__ int calcH(int x, int y, int lenX, int lenY) {
	int r = 0;
	if (x < lenX - 1 && y < lenY - 1)
		r = 1;
	return r;
}

__global__ void yeeKernel(int lenX, int lenY, bool h, int k, double CEy, double CEx, double CH, double * Ez, double * Hx, double * Hy, int * boundEz, int * boundHx, int * boundHy) {
	// Reconhecimento espacial.
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= lenX || y >= lenY)
		return;

	int gw = lenX;
	int gh = lenY;
	int is = (y * gw) + x;

	int doE = calcE(x, y, lenX, lenY);
	int doH = calcH(x, y, lenX, lenY);

	bool bEz = boundEz[is] + doE == 2;
	bool bHx = boundHx[is] + doH == 2;
	bool bHy = boundHy[is] + doH == 2;
	double _hy, _hx, _ez;

	if (k==0) { _hy = 0; _hx = 0; }
	else { _hy = Hy[iS(x,y,gw)]; _hx = Hx[iS(x,y,gw)]; }
	_ez = Ez[iST(x,y,k,gw,gh)];

	if (h) {
	// Calcula HX
		if (bHx) {
			_hx = _hx - CH * (Ez[iST(x+1,y,k,gw,gh)] - _ez);
			Hx[is] = _hx;
		}
		// Calcula HY
		if (bHy) {
			_hy = _hy + CH * (Ez[iST(x,y+1,k,gw,gh)] - _ez);
			Hy[is] = _hy;
		}
	}

	else {
	// Calcula Ez
		if (bEz) {
			_ez = _ez + (CEx * ( _hy - Hy[iS(x,y-1,gw)] )) - (CEy * (_hx - Hx[iS(x-1,y,gw)]));
			Ez[iST(x,y,k+1,gw,gh)] = _ez;
		}
	}

	return;
}


extern "C" int run(int lenX, int lenY, int lenT, double CEy, double CEx, double CH,  double * Ez, int * boundEz, int * boundHx, int * boundHy) {

	// Mallocs
	double * dEz, * dHx, * dHy;
	size_t s_ez = sizeof(double)*lenX*lenY*lenT;
	CudaSafeCall(cudaMalloc(&dEz, s_ez));
	CudaSafeCall(cudaMalloc(&dHx, sizeof(double)*lenX*lenY));
	CudaSafeCall(cudaMalloc(&dHy, sizeof(double)*lenX*lenY));

	int * dbEz, * dbHx, * dbHy;
	size_t bs = sizeof(int)*lenX*lenY;
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
		yeeKernel<<<blocks, threads>>>(lenX, lenY, true, k, CEy, CEx, CH, dEz, dHx, dHy, dbEz, dbHx, dbHy);
		yeeKernel<<<blocks, threads>>>(lenX, lenY, false, k, CEy, CEx, CH, dEz, dHx, dHy, dbEz, dbHx, dbHy);
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
