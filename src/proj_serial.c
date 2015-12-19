#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/times.h>
#include <sys/time.h>
#include "fft.c"

#define N 512

void readInputFile(char* fileName, complex **data)
{
  FILE *fp = fopen(fileName, "rb");
  int i, j;

  for (i=0;i<N;i++){
    for (j=0;j<N;j++){
      float x;
      fscanf(fp,"%f",&x);
      data[i][j].r = x;
      data[i][j].i = 0.0;
    }
  }
  fclose(fp);
}

void fft2d(complex **data, int isign)
{
  int i, j;
  complex *d1;

  /* Transform the rows */
  d1 = malloc(N * sizeof(complex *));
  for(i = 0; i < N; i++) {
    for(j = 0; j < N; j++) {
      d1[j].r = data[i][j].r;
      d1[j].i = data[i][j].i;
    }
    c_fft1d(d1, N, isign);
    for(j = 0; j < N; j++) {
      data[i][j].r = d1[j].r;
      data[i][j].i = d1[j].i;
    }
  }
  free(d1);

  /* Transform the columns */
  d1 = malloc(N * sizeof(complex *));
  for(i = 0; i < N; i++) {
    for(j = 0; j < N; j++) {
      d1[j].r = data[j][i].r;
      d1[j].i = data[j][i].i;
    }
    c_fft1d(d1, N, isign);
    for(j = 0; j < N; j++) {
      data[j][i].r = d1[j].r;
      data[j][i].i = d1[j].i;
    }
  }
  free(d1);
}

void multiplication(complex **A, complex **B, complex **C)
{
    int i, j;

    for(i=0;i<N;i++){
        for(j=0;j<N;j++){
            C[i][j].r = (A[i][j].r * B[i][j].r) - (A[i][j].i * B[i][j].i);
            C[i][j].i = (A[i][j].r * B[i][j].i) + (A[i][j].i * B[i][j].r);
        }
    }
}

void writeOutputFile(char* fileName, complex **data)
{
    FILE *fp = fopen(fileName, "wb");
    int i, j;

    for (i=0;i<N;i++) {
        for (j=0;j<N;j++){
            fprintf(fp,"%.7e\t",data[i][j].r);
        }
        fprintf(fp,"\n");
    }
    fclose(fp);
}

int main()
{
  struct timeval etstart, etstop;  /* Elapsed times using gettimeofday() */
  struct timezone tzdummy;
  clock_t etstart2, etstop2;  /* Elapsed times using times() */
  unsigned long long usecstart, usecstop;
  struct tms cputstart, cputstop;  /* CPU times for my processes */

  complex **A, **B, **C;
  int i;

  A = malloc(N * sizeof(complex *));
  B = malloc(N * sizeof(complex *));
  C = malloc(N * sizeof(complex *));

  for(i = 0; i < N; i++) {
    A[i] = malloc(N * sizeof(complex *));
    B[i] = malloc(N * sizeof(complex *));
    C[i] = malloc(N * sizeof(complex *));
  }

  char *input1 = "sample/1_im1";
  char *input2 = "sample/1_im2";
  char *output = "out_test1";

  readInputFile(input1, A);
  readInputFile(input2, B);

  /* Start Clock */
  printf("\nStarting clock.\n");
  gettimeofday(&etstart, &tzdummy);
  etstart2 = times(&cputstart);

  fft2d(A, -1);
  fft2d(B, -1);

  multiplication(A, B, C);

  fft2d(C, 1);

  /* Stop Clock */
  gettimeofday(&etstop, &tzdummy);
  etstop2 = times(&cputstop);
  printf("Stopped clock.\n");
  usecstart = (unsigned long long)etstart.tv_sec * 1000000 + etstart.tv_usec;
  usecstop = (unsigned long long)etstop.tv_sec * 1000000 + etstop.tv_usec;

  printf("\nElapsed time = %g ms.\n", (float)(usecstop - usecstart)/(float)1000);

  writeOutputFile(output, C);

  free(A);
  free(B);
  free(C);
  return 0;
}
