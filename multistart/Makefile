CC=gcc
CC_MPI=mpicc
CFLAGS=-Wall -O3
OMP_FLAGS=-fopenmp
DEBUG_FLAGS=-DDEBUG 
LDLIBS=-lm

#TODO: add the following cases: multistart_mds_omp multistart_mds_omp_tasks multistart_mds_mpi

all: multistart_mds_seq multistart_mds_openMP multistart_mds_omp_tasks multistart_mds_mpi

multistart_mds_seq: multistart_mds_seq.c torczon.c
	$(CC) -Wall -O3 -o multistart_mds_seq multistart_mds_seq.c torczon.c $(LDLIBS)

multistart_mds_openMP: multistart_mds_openMP.c torczon.c
	$(CC) $(CFLAGS) $(OMP_FLAGS) -o multistart_mds_openMP multistart_mds_openMP.c torczon.c $(LDLIBS)
	
multistart_mds_omp_tasks: multistart_mds_omp_tasks.c torczon_openMP_tasks.c
        $(CC) $(CFLAGS) $(OMP_FLAGS) -o multistart_mds_omp_tasks multistart_mds_omp_tasks.c torczon_openMP_tasks.c $(LDLIBS)

multistart_mds_mpi: multistart_mds_mpi.c torczon.c
	$(CC_MPI) $(CFLAGS) -o multistart_mds_mpi multistart_mds_mpi.c torczon.c $(LDLIBS)

clean:
	rm -f multistart_mds_seq multistart_mds_omp multistart_mds_omp_tasks multistart_mds_mpi
