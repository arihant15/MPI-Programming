
CC := mpicc
GC := gcc

assignment2: $(OBJ)
	$(GC) -o proj_serial proj_serial.c -lm
	$(CC) -c proj_mpi.c
	$(CC) -o proj_mpi proj_mpi.o -lm
	$(CC) -c proj_mpi_collective.c
	$(CC) -o proj_mpi_collective proj_mpi_collective.o -lm
	$(CC) -c proj_mpi_omp.c
	$(CC) -o proj_mpi_omp proj_mpi_omp.o -fopenmp -lm
	$(CC) -c proj_mpi_taskdata_parallel.c
	$(CC) -o proj_mpi_taskdata_parallel proj_mpi_taskdata_parallel.o -lm

q1:
	./proj_serial

q2:
	mpirun -n 8 ./proj_mpi

q3:
	mpirun -n 8 ./proj_mpi_collective

q4:
	mpirun -n 8 ./proj_mpi_omp

q5:
	mpirun -n 8 ./proj_mpi_taskdata_parallel

clean:
	rm -rf proj_serial proj_mpi proj_mpi_collective proj_mpi_omp proj_mpi_taskdata_parallel *out* *.o
