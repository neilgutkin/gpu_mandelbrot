CUDA-based Mandelbrot generator with 2D threading

Generates an image of the Mandelbrot set at selected coordinates and times kernel execution (printed to stdout).

To compile:
nvcc -O3 -o mandelbrot mandelbrot.cu png_util.c -I. -lm -lpng

To run:
./mandelbrot

Learn more about the Mandelbrot set: https://en.wikipedia.org/wiki/Mandelbrot_set
