//#define BGQ 1 // when running BG/Q, comment out when running on mastiff
#include <mpi.h>
#ifdef BGQ
#include<hwi/include/bqc/A2_inlines.h>
#else
#define GetTimeBase MPI_Wtime
#endif
double processor_frequency = 1600000000.0;

void MPI_P2P_Reduce() {
}

int main(int argc, char* argv[]) {
	unsigned long long start_cycles = GetTimeBase();
	MPI_P2P_Reduce();
	unsigned long long end_cycles = GetTimeBase();
	double time_in_secs = ((double)(end_cycles - start_cycles)) / processor_frequency;

}
