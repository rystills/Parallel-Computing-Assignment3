//#define BGQ 1 // when running BG/Q, comment out when running on mastiff
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <mpi.h>
#include <stdbool.h>
#include <math.h>
#ifdef BGQ
#include<hwi/include/bqc/A2_inlines.h>
#else
#define GetTimeBase MPI_Wtime
#endif
#define DEBUG false

//MPI data
double processor_frequency = 1600000000.0;
int numRanks = -1; //total number of ranks in the current run
int rank = -1; //our rank

void MPI_P2P_Reduce() {
	if (rank == 0) {
		printf("%llu\n",(unsigned long long)(pow(2,29)*(pow(2,30)-1)));
	}
}

int main(int argc, char* argv[]) {
	//usage check
	if (argc != 2) {
		fprintf(stderr,"Error: %d input argument[s] supplied, but 1 was expected. Usage: mpirun -np X ./p2p.out output.txt\n",argc-1);
		exit(1);
	}

	//init MPI + grab size and rank
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	//only rank needs to open the output file for writing
	if (rank == 0) {
		freopen(argv[1], "w", stdout);
	}

	//everyone runs the main routine
	unsigned long long start_cycles = GetTimeBase();
	MPI_P2P_Reduce();

	//time measurement, for analysis during test runs
	if (DEBUG) {
		fprintf(stderr, "Execution time: %f",((double)(GetTimeBase() - start_cycles)) / processor_frequency);
	}

	//all done
	MPI_Finalize();
}
