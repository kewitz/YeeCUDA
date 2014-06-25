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
	unsigned int lenX, lenY, lenT;
	double Z;
	double CEy;
	double CEx;
	double CH;
};
// #define iS (x, y, lenx) ((lenx*y) + x)
// #define iST (x, y, k, lenx, leny) ((lenx*leny*k) + (lenx*y) + x)
// Global index in Space domain.
__device__ __inline__ uint iS(unsigned int x, unsigned int y, unsigned int lenx, unsigned int leny){
	uint p = (y * lenx) + x;
	return p;
}
// Global index in Space-Time domain.
__device__ __inline__ uint iST(unsigned int x, unsigned int y, unsigned int k, unsigned int lenx, unsigned int leny){
	uint p = (k*lenx*leny) + iS(x,y,lenx,leny);
	return p;
}

__device__ __inline__ int calcE(Yee s, uint3 gp) {
	int r = 0;
	if (gp.x > 0 && gp.y > 0 && gp.x < s.lenX - 1 && gp.y < s.lenY -1)
		r = 1;
	return r;
}
__device__ __inline__ int calcH(Yee s, uint3 gp) {
	int r = 0;
	if (gp.x < s.lenX - 1 && gp.y < s.lenY - 1)
		r = 1;
	return r;
}

__global__ void yeeKernel(Yee s, int ks, double * Ez, double * Hx, double * Hy, int * boundEz, int * boundHx, int * boundHy) {
	// Reconhecimento espacial.
	uint3 gp = __getGlobalPosition(blockIdx, blockDim, threadIdx);
	if (gp.x >= s.lenX || gp.y >= s.lenY)
		return;

	unsigned int is = iS(gp.x, gp.y, s.lenX, s.lenY);

	int doE = calcE(s, gp);
	int doH = calcH(s, gp);

	bool bEz = boundEz[is] + doE == 2;
	bool bHx = boundHx[is] + doH == 2;
	bool bHy = boundHy[is] + doH == 2;

	unsigned int k,x=gp.x,y=gp.y;
	double _hy = 0.0, _hx = 0.0, _ez = Ez[iST(x,y,0,s.lenX,s.lenY)];

	for(k = 0; k < ks-1; k++)
	{
		__syncthreads();
		// Calcula HX
		if (bHx) {
			_hx = _hx - s.CH * (Ez[iST(x+1,y,k,s.lenX,s.lenY)] - _ez);
			Hx[is] = _hx;
		}
		// Calcula HY
		if (bHy) {
			_hy = _hy + s.CH * (Ez[iST(x,y+1,k,s.lenX,s.lenY)] - _ez);
			Hy[is] = _hy;
		}
		__syncthreads();

		// Calcula Ez
		if (bEz) {
			_ez = _ez + (s.CEx * ( _hy - Hy[iS(x,y-1,s.lenX,s.lenY)] )) - (s.CEy * (_hx - Hx[iS(x-1,y,s.lenX,s.lenY)]));
			Ez[iST(x,y,k+1,s.lenX,s.lenY)] = _ez;
		}
		__syncthreads();

	}

}


extern "C" int run(Yee s, int iter, double * Ez, int * boundEz, int * boundHx, int * boundHy) {

	// Mallocs
	double * dEz, * dHx, * dHy;
	size_t s_ez = sizeof(double)*s.lenX*s.lenY*s.lenT;
	CudaSafeCall(cudaMalloc(&dEz, s_ez));
	CudaSafeCall(cudaMalloc(&dHx, sizeof(double)*s.lenX*s.lenY));
	CudaSafeCall(cudaMalloc(&dHy, sizeof(double)*s.lenX*s.lenY));

	int * dbEz, * dbHx, * dbHy;
	size_t bs = sizeof(int)*s.lenX*s.lenY;
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
	const dim3 blocks(1 + s.lenX/threads.x, 1 + s.lenY/threads.y);
	yeeKernel<<<blocks, threads>>>(s, iter, dEz, dHx, dHy, dbEz, dbHx, dbHy);
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

void test(){
	Yee p = {10,10,5,376.730313475,188.365156737,188.365156737,0.00132720936467};
	double ez[500] = {};
	int bound[100] = {};
	for(int k = 0; k < 100; k++)
		bound[k] = 1;

	run(p,2,&ez[0],&bound[0],&bound[0],&bound[0]);
}

int main(){
	int theresCuda = checkCuda();
	unsigned int a = 0;
	int p = a+theresCuda;
	return theresCuda;
}
