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

void fft1dRow(complex **data, int isign, int lowerBound, int upperBound)
{
  int i, j;
  complex *d1;

  d1 = (complex *)malloc(N * sizeof(complex));
  for(i = lowerBound; i < upperBound; i++) {
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

void fft2d(complex **data, int isign, int my_rank, int p)
{
  int i, j;
  complex *d1;

  /* Transform the rows */
  d1 = malloc(N * sizeof(complex *));
  for(i = my_rank; i < N; i+=p) {
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
  for(i = my_rank; i < N; i+=p) {
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

  A = malloc(N * sizeof(complex *));
  B = malloc(N * sizeof(complex *));
  C = malloc(N * sizeof(complex *));
  A1 = malloc(N * sizeof(complex *));
  B1 = malloc(N * sizeof(complex *));
  C1 = malloc(N * sizeof(complex *));

  for(i = 0; i < N; i++) {
    A[i] = malloc(N * sizeof(complex *));
    B[i] = malloc(N * sizeof(complex *));
    C[i] = malloc(N * sizeof(complex *));
    A1[i] = malloc(N * sizeof(complex *));
    B1[i] = malloc(N * sizeof(complex *));
    C1[i] = malloc(N * sizeof(complex *));
  }

  char* input1 = "sample/1_im1";
  char* input2 = "sample/1_im2";
  char* output = "mpi_out_test1";

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

  int workload = N/p;
  int lowerBound = my_rank * workload;
  int upperBound = lowerBound + workload;
  int offset;

  // printf("\nTransfering data...");
  if(my_rank == 0) {
    readInputFile(input1, A);
    readInputFile(input2, B);

    printf("\nStarting Timer...");
    startTime = MPI_Wtime();

    for (i = 1; i < p; i++) {
      offset = i * workload;
      for(j = offset; j < (offset+workload); j++) {
        MPI_Send(&A[j][0], N, mpi_complex, i, 0, MPI_COMM_WORLD);
        MPI_Send(&B[j][0], N, mpi_complex, i, 0, MPI_COMM_WORLD);
      }
    }
  }
  else {
    for(j = lowerBound; j < upperBound; j++) {
      MPI_Recv(A[j], N, mpi_complex, 0, 0, MPI_COMM_WORLD, &status);
      MPI_Recv(B[j], N, mpi_complex, 0, 0, MPI_COMM_WORLD, &status);
    }
  }

  // printf("\nStarting fft1d Row on A & B...");
  fft1dRow(A, -1, lowerBound, upperBound);
  fft1dRow(B, -1, lowerBound, upperBound);

  if (my_rank != 0) {
    for(i = lowerBound; i < upperBound; i++) {
      MPI_Send(&A[i][0], N, mpi_complex, 0, 0, MPI_COMM_WORLD);
      MPI_Send(&B[i][0], N, mpi_complex, 0, 0, MPI_COMM_WORLD);
    }
  }
  else {
    for(i = 1; i < p; i++) {
      offset = i * workload;
      for(j = offset; j < (offset+workload); j++) {
        MPI_Recv(A[j], N, mpi_complex, i, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(B[j], N, mpi_complex, i, 0, MPI_COMM_WORLD, &status);
      }
    }
  }

  if(my_rank == 0) {
    // printf("\nTransposing the matrix...");
    transpose(A,A1);
    transpose(B,B1);

    for(i = 1; i < p; i++) {
      offset = i * workload;
      for(j = offset; j < offset+workload; j++) {
        MPI_Send(&A1[j][0], N, mpi_complex, i, 0, MPI_COMM_WORLD);
        MPI_Send(&B1[j][0], N, mpi_complex, i, 0, MPI_COMM_WORLD);
      }
    }
  }
  else {
    for(j = lowerBound; j < upperBound; j++) {
      MPI_Recv(A1[j], N, mpi_complex, 0, 0, MPI_COMM_WORLD, &status);
      MPI_Recv(B1[j], N, mpi_complex, 0, 0, MPI_COMM_WORLD, &status);
    }
  }

  // printf("\nStarting fft1d Row on A & B...");
  fft1dRow(A1, -1, lowerBound, upperBound);
  fft1dRow(B1, -1, lowerBound, upperBound);

  if (my_rank != 0) {
    for(i = lowerBound; i < upperBound; i++) {
      MPI_Send(&A1[i][0], N, mpi_complex, 0, 0, MPI_COMM_WORLD);
      MPI_Send(&B1[i][0], N, mpi_complex, 0, 0, MPI_COMM_WORLD);
    }
  }
  else {
    for(i = 1; i < p; i++) {
      offset = i * workload;
      for(j = offset; j < offset+workload; j++) {
        MPI_Recv(A1[j], N, mpi_complex, i, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(B1[j], N, mpi_complex, i, 0, MPI_COMM_WORLD, &status);
      }
    }
  }

  if (my_rank == 0) {
    transpose(A1,A);
    transpose(B1,B);
    // printf("\nStarting M*M point...");
    multiplication(A, B, C);

    for(i = 1; i < p; i++) {
      offset = i * workload;
      for(j = offset; j < offset+workload; j++) {
        MPI_Send(&C[j][0], N, mpi_complex, i, 0, MPI_COMM_WORLD);
      }
    }
  }
  else
  {
    for(i = lowerBound; i < upperBound; i++) {
      MPI_Recv(C[i], N, mpi_complex, 0, 0, MPI_COMM_WORLD, &status);
    }
  }

  // printf("\nStarting fft1d Row on C...");
  fft1dRow(C, 1, lowerBound, upperBound);

  if (my_rank != 0) {
    for(i = lowerBound; i < upperBound; i++) {
      MPI_Send(&C[i][0], N, mpi_complex, 0, 0, MPI_COMM_WORLD);
    }
  }
  else {
    for(i = 1; i < p; i++) {
      offset = i * workload;
      for(j = offset; j < offset+workload; j++) {
        MPI_Recv(C[j], N, mpi_complex, i, 0, MPI_COMM_WORLD, &status);
      }
    }
  }

  if(my_rank == 0) {
    // printf("\nTransposing the matrix...");
    transpose(C,C1);
    for(i = 1; i < p; i++) {
      offset = i * workload;
      for(j = offset; j < offset+workload; j++) {
        MPI_Send(&C1[j][0], N, mpi_complex, i, 0, MPI_COMM_WORLD);
      }
    }
  }
  else {
    for(i = lowerBound; i < upperBound; i++) {
      MPI_Recv(C1[i], N, mpi_complex, 0, 0, MPI_COMM_WORLD, &status);
    }
  }

  // printf("\nStarting fft1d Row on C...");
  fft1dRow(C1, 1, lowerBound, upperBound);

  if (my_rank != 0) {
    for(i = lowerBound; i < upperBound; i++) {
      MPI_Send(&C1[i][0], N, mpi_complex, 0, 0, MPI_COMM_WORLD);
    }
  }
  else {
    for(i = 1; i < p; i++) {
      offset = i * workload;
      for(j = offset; j < offset+workload; j++) {
        MPI_Recv(C1[j], N, mpi_complex, i, 0, MPI_COMM_WORLD, &status);
      }
    }
  }
  
  if(my_rank == 0) {
    transpose(C1,C);
    endTime = MPI_Wtime();
    printf("\nWriting output...\n");
    writeOutputFile(output, C);
    printf("\nElapsed time = %lf s.\n",(endTime - startTime));
    printf("--------------------------------------------\n");
  }

  free(A);
  free(B);
  free(C);
  free(A1);
  free(B1);
  free(C1);
  MPI_Finalize();
  return 0;
}
