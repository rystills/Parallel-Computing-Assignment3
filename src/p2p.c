#define BGQ 1 // when running BG/Q, comment out when running on mastiff
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <mpi.h>
#include <stdbool.h>
#include <math.h>
#ifdef BGQ
#include<hwi/include/bqc/A2_inlines.h>
#define processor_frequency 1600000000.0
#else
#define GetTimeBase MPI_Wtime
#define processor_frequency 1.0
#endif
#define DEBUG true
#define numEntries 1073741824

//MPI data
int numRanks = -1; //total number of ranks in the current run
int rank = -1; //our rank
int elementsPerProc = -1; //stores how many elements each process is responsible for

/**
 * apply a point to point reduction on the send_data, storing the final result in recv_data
 * @param send_data: array of elements of type datatype that we are going to reduce
 * @param recv_data: element of type datatype in which we will store the reduction result
 * @param count: number of elements in send_data
 * @param datatype: data type of elements in send_data (hard-coded unsigned long as per spec)
 * @param op: operation to be performed (hard-coded MPI_SUM as per spec)
 * @param root: the process on which the final value will be stored (hard-coded rank 0 as per spec)
 * @param communicator: process grouping communicator to use (hard-coded MPI_COMM_WORLD as per spec)
 */
void MPI_P2P_Reduce(unsigned long* send_data, unsigned long long* recv_data, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm communicator) {
	//calculate this rank's sum
	*recv_data = 0;
	for (unsigned long i = 0; i < count; *recv_data += send_data[i++]);
	//pairwise add across ranks
	MPI_Request request;
	MPI_Status status;
	unsigned long long buffer;
	for (int i = 1; i <= numRanks/2; i*=2) {
		//receiver
		if (rank % (i*2) == 0) {
			MPI_Irecv(&buffer, 1, MPI_UNSIGNED_LONG_LONG, rank+i, 0, MPI_COMM_WORLD, &request);
		}
		//sender
		else {
			MPI_Isend(recv_data, 1, MPI_UNSIGNED_LONG_LONG, rank-i, 0, MPI_COMM_WORLD, &request);
			return;
		}
		MPI_Wait(&request, &status);
		*recv_data += buffer;
	}
}

int main(int argc, char* argv[]) {
	//usage check
	if (argc != 1) {
		fprintf(stderr,"Error: %d input argument[s] were supplied, but 0 were expected. Usage: mpirun -np X ./p2p.out\n",argc-1);
		exit(1);
	}

	//init MPI & get size, rank, and elementsPerProc
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	elementsPerProc = numEntries/(float)numRanks;

	//populate deterministic input data
	unsigned long *inputData = malloc(elementsPerProc * sizeof(unsigned long));
	for (unsigned long i = 0; i < elementsPerProc; ++i) {
		inputData[i] = rank*elementsPerProc + i;
	}
	unsigned long long localSum = 0, recv_sum = 0;

	//execute point to point reduction
	unsigned long long start_cycles = GetTimeBase();
	MPI_P2P_Reduce(inputData, &recv_sum, elementsPerProc, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
	double time_in_secs = (GetTimeBase() - start_cycles) / processor_frequency;
	//only rank 0 needs to write the output
	if (!rank) {
		printf("%llu%s",recv_sum, DEBUG ? " " : "\n");
		if (DEBUG) fprintf(stdout, "P2P Execution time: %fs\n",time_in_secs);
	}

	//execute collective reduction
	start_cycles = GetTimeBase();
	//sum the elements locally on each rank so that we don't have to allocate an additional temp array for MPI_SUM result storage
	for (unsigned long i = 0; i < elementsPerProc; localSum += inputData[i++]);
	MPI_Reduce(&localSum, &recv_sum, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
	time_in_secs = (GetTimeBase() - start_cycles) / processor_frequency;
	//only rank 0 needs to write the output
	if (!rank) {
		printf("%llu%s",recv_sum, DEBUG ? " " : "\n");
		if (DEBUG) fprintf(stdout, "COL Execution time: %fs\n",time_in_secs);
	}

	//all done
	MPI_Finalize();
	free(inputData);
}
