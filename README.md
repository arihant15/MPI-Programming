# MPI-Programming
MPI-Programming

Compilation Steps:

	Method 1: Run make command.
	
		eg: $ make

	Method 2: compile each code individually.

	Sequential Code:
		$gcc -o proj_serial proj_serial.c -lm
		
	MPI Send and Recv Code:
		$mpicc -c proj_mpi.c
		$mpicc -o proj_mpi proj_mpi.o -lm
		
	MPI Collective Code:
		$mpicc -c proj_mpi_collective.c
		$mpicc -o proj_mpi_collective proj_mpi_collective.o -lm
		
	MPI and openMP Code:
		$mpicc -c proj_mpi_omp.c
		$mpicc -o proj_mpi_omp proj_mpi_omp.o -fopenmp -lm
		
	MPI Task and Data Parallel:
		$mpicc -c proj_mpi_taskdata_parallel.c
		$mpicc -o proj_mpi_taskdata_parallel proj_mpi_taskdata_parallel.o -lm

	Note: Ignore warning
Execution steps:

	Sequential Code:			$./proj_serial 				or 		$make q1
	MPI Send and Recv Code:		$mpirun -n 8 ./proj_mpi		or		$make q2
	MPI Collective Code:		$mpirun -n 8 ./proj_mpi_collective		or 		$make q3
	MPI and openMP Code:		$mpirun -n 8 ./proj_mpi_omp			or 			$make q4
	MPI Task and Data Parallel:		$mpirun -n 8 ./proj_mpi_taskdata_parallel		or 		$make q5

Output Files:

	Sequential Code:	out_test1
	MPI Send and Recv Code:	mpi_out_test1
	MPI Collective Code:	mpi_collective_out_test1
	MPI and openMP Code:	mpi_omp1_out_test1
	MPI Task and Data Parallel:	mpi_taskdata_out_test1
