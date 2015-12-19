#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
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
      data[i][j].i = 0.00;
    }
  }
  fclose(fp);
}

void fft1dRow(complex **data, int isign, int my_rank, int offset)
{
  int i, j;
  complex *d1;

  d1 = (complex *)malloc(N * sizeof(complex));
  for(i = my_rank*offset; i < (N/2+(my_rank*offset)); i++) {
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

void transpose(complex **data, complex **trans) 
{
  int i,j;
  for(i = 0; i < N; i++) {
    for(j = 0; j < N; j++) {
      trans[j][i] = data[i][j];
    }
  }
}

void multiplication(complex **A, complex **B, complex **C)
{
    int i, j;

    for(i=0;i<N;i++){
        for(j=0;j<N;j++){
          //printf("A.r=%f,A.i=%f,B.r=%f,B.i=%f",A[i][j].r,A[i][j].i,B[i][j].r,B[i][j].i);
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

int malloc2dcomplex(complex ***array, int n, int m) {

    /* allocate the n*m contiguous items */
    complex *p = (complex *)malloc(n*m*sizeof(complex));
    if (!p) return -1;

    /* allocate the row pointers into the memory */
    (*array) = (complex **)malloc(n*sizeof(complex*));
    if (!(*array)) {
       free(p);
       return -1;
    }

    /* set up the pointers into the contiguous memory */
    int i;
    for (i=0; i<n; i++) 
       (*array)[i] = &(p[i*m]);

    return 0;
}

int free2dcomplex(complex ***array) {
    /* free the memory - the first element of the array is at the start */
    free(&((*array)[0][0]));

    /* free the pointers into the memory */
    free(*array);

    return 0;
}

int main(int argc, char **argv)
{
  complex **A, **B, **C, **A1, **B1, **C1;
  int i, j;
  int my_rank;
  int p;
  int source;
  int dest;
  double startTime,endTime;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);
  MPI_Status status;

  malloc2dcomplex(&A, N, N);
  malloc2dcomplex(&B, N, N);
  malloc2dcomplex(&C, N, N);
  malloc2dcomplex(&A1, N, N);
  malloc2dcomplex(&B1, N, N);
  malloc2dcomplex(&C1, N, N);

  char* input1 = "sample/1_im1";
  char* input2 = "sample/1_im2";
  char* output = "mpi_taskdata_out_test1";

  /* MPI DataType */
  int count = 2;
  int lengths[2] = {1, 1};
  MPI_Aint offsets[2] = {0, sizeof(float)};
  MPI_Datatype types[2] = {MPI_FLOAT, MPI_FLOAT};
  MPI_Datatype mpi_complex;
  MPI_Type_struct(count, lengths, offsets, types, &mpi_complex);
  MPI_Type_commit(&mpi_complex);

  /* Processing */
  printf("\nMy rank: %d, total processors: %d", my_rank, p);
  
  if(my_rank == 0) {
    readInputFile(input1, A);
    
    fft2d(A, -1);

    MPI_Send(&A[0][0], N * N, mpi_complex, 2, 0, MPI_COMM_WORLD);
    
  }
  else if(my_rank == 1) {
    readInputFile(input2, B);
    
    fft2d(B, -1);

    MPI_Send(&B[0][0], N * N, mpi_complex, 2, 0, MPI_COMM_WORLD);
  }
  else if(my_rank == 2) {
    MPI_Recv(&A[0][0], N * N, mpi_complex, 0, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(&B[0][0], N * N, mpi_complex, 1, 0, MPI_COMM_WORLD, &status);
    
    multiplication(A, B, C);
    
    MPI_Send(&C[0][0], N * N, mpi_complex, 3, 0, MPI_COMM_WORLD);
  }
  else if(my_rank == 3) {
    printf("\nStarting Timer...");
    startTime = MPI_Wtime();

    MPI_Recv(&C[0][0], N * N, mpi_complex, 2, 0, MPI_COMM_WORLD, &status);
    
    fft2d(C, 1);
    
    endTime = MPI_Wtime();
    printf("\nWriting output...\n");
    writeOutputFile(output, C);
    printf("\nElapsed time = %lf s.\n",(endTime - startTime));
    printf("--------------------------------------------\n");
  }

  free2dcomplex(&A);
  free2dcomplex(&B);
  free2dcomplex(&C);
  free2dcomplex(&A1);
  free2dcomplex(&B1);
  free2dcomplex(&C1);
  MPI_Finalize();
  return 0;
}