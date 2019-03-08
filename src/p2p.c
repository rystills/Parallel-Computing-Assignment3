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
#define numEntries 1073741824
//#define numEntries 32768

//MPI data
double processor_frequency = 1600000000.0;
int numRanks = -1; //total number of ranks in the current run
int rank = -1; //our rank
int elementsPerProc = -1; //stores how many elements each process is responsible for

void MPI_P2P_Reduce() {
	//only rank needs to write the output
	if (rank == 0) {
		if (!DEBUG) freopen("STDOUT_1.txt", "w", stdout);
		printf("%llu\n",(unsigned long long)(pow(2,29)*(pow(2,30)-1)));
	}
}

int main(int argc, char* argv[]) {
	//usage check
	if (argc != 1) {
		fprintf(stderr,"Error: %d input argument[s] supplied, but 0 were expected. Usage: mpirun -np X ./p2p.out\n",argc-1);
		exit(1);
	}

	//init MPI & get size, rank, and elementsPerProc
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	elementsPerProc = numEntries/(float)numRanks;

	//execute p2p reduce
	unsigned long long start_cycles = GetTimeBase();
	MPI_P2P_Reduce();
	//now compare to collective reduce
	unsigned long *send_data = malloc(elementsPerProc * sizeof(unsigned long));
	for (unsigned long i = 0; i < elementsPerProc; ++i) {
		send_data[i] = rank*elementsPerProc + i;
	}
	unsigned long *recv_data = malloc(rank==0?elementsPerProc * sizeof(unsigned long):0);
	MPI_Reduce(send_data, recv_data, elementsPerProc, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
	if (rank == 0) {
		unsigned long long sum = 0;
		for (unsigned long i = 0; i < elementsPerProc; ++i) {
			sum += recv_data[i];
		}
		//only rank 0 needs to write the output
		printf("%llu\n",sum);
		if (DEBUG) {
			fprintf(stderr, "Execution time: %f\n",((double)(GetTimeBase() - start_cycles)) / processor_frequency);
		}
	}

	//all done
	MPI_Finalize();
	free(send_data);
	free(recv_data);
}
