// Neil Gutkin
// 09/29/21

/*

To compile:
nvcc -O3 -o mandelbrot mandelbrot.cu png_util.c -I. -lm -lpng

To run:
./mandelbrot

*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

// include as a C file (no name mangling)
extern "C" {
  #include "png_util.h"
}

/*
z <- z^2 + c
Perform the above iteration for each complex value c and count
how many iterations it takes before the magnitude of the complex number
z satisfies |z|<4
*/
__global__ void mandelbrotKernel(const int NRe, 
		const int NIm, 
		const float minRe,
		const float minIm,
		const float dRe, 
		const float dIm,
		float * h_count){

  // orient thread
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int dx = blockDim.x;
  
  int ty = threadIdx.y;
  int by = blockIdx.y;
  int dy = blockDim.y;
  
  // establish which entry this thread is responsible for
  int column = tx + bx*dx;
  int row    = ty + by*dy;

  if (row < NIm && column < NRe) { // bounds check
      float cRe = minRe + column*dRe;
      float cIm = minIm + row*dIm;

      float zRe = 0;
      float zIm = 0;
      
      int Nt = 200;
      int t, cnt=0;
      for(t=0;t<Nt;++t){
	
	// z = z^2 + c
	//   = (zRe + i*zIm)*(zRe + i*zIm) + (cRe + i*cIm)
	//   = zRe^2 - zIm^2 + 2*i*zIm*zRe + cRe + i*cIm
	      float zReTmp = zRe*zRe - zIm*zIm + cRe;
	      zIm = 2.f*zIm*zRe + cIm;
	      zRe = zReTmp;

	      cnt += (zRe*zRe+zIm*zIm<4.f);
      }

      h_count[column + row*NRe] = cnt;
  }

}


int main(int argc, char **argv){

  const int NRe = 4096;
  const int NIm = 4096;

  // box containing sample points 
  const float centRe = -1.2, centIm= -.2;
  const float diam  = 0.3;
  const float minRe = centRe-0.5*diam;
  const float remax = centRe+0.5*diam;
  const float minIm = centIm-0.5*diam;
  const float immax = centIm+0.5*diam;

  const float dRe = (remax-minRe)/(NRe-1.f);
  const float dIm = (immax-minIm)/(NIm-1.f);

  // allocate HOST array
  float *h_count = (float*) calloc(NRe*NIm, sizeof(float));

  // allocate DEVICE array
  float* c_count;
  cudaMalloc(&c_count, NRe*NIm * sizeof(float));

  // establish threading dimensions
  dim3 B(16,16,1);
  dim3 G((NIm+16-1)/16, (NRe+16-1)/16, 1);

  // warm up kernel
  mandelbrotKernel <<< G , B >>> (NRe, NIm, minRe, minIm, dRe, dIm, c_count);

  // set up timing
  cudaEvent_t tic, toc;
  cudaEventCreate(&tic);
  cudaEventCreate(&toc);

  // start timer
  cudaDeviceSynchronize();
  cudaEventRecord(tic);

  // call mandelbrot from here
  mandelbrotKernel <<< G , B >>> (NRe, NIm, minRe, minIm, dRe, dIm, c_count);

  // end timer
  cudaEventRecord(toc);
  cudaDeviceSynchronize();

  // get and print elapsed time
  float elapsed;
  cudaEventElapsedTime(&elapsed, tic, toc);
  elapsed /= 1000;
  
  printf("elapsed time %f seconds\n", elapsed);

  // copy results from DEVICE to HOST
  cudaMemcpy(h_count, c_count, NRe*NIm * sizeof(float), cudaMemcpyDeviceToHost);

  // create the png
  FILE *png = fopen("mandelbrot.png", "w");
  write_hot_png(png, NRe, NIm, h_count, 0, 80);
  fclose(png);

}
