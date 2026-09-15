/* Four-node CPU MPI smoke test: membership, ring exchange, and reduction. */
#include <mpi.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv)
{
    int rank, size, name_len;
    char host[MPI_MAX_PROCESSOR_NAME] = {0};
    char hosts[4][MPI_MAX_PROCESSOR_NAME] = {{0}};
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 4) {
        if (rank == 0) fprintf(stderr, "FAIL: expected 4 ranks, got %d\n", size);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    MPI_Get_processor_name(host, &name_len);
    MPI_Allgather(host, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                  hosts, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, MPI_COMM_WORLD);
    int errors = 0;
    if (rank == 0) {
        for (int i = 0; i < size; ++i) {
            printf("rank %d/%d on %s\n", i, size, hosts[i]);
            for (int j = 0; j < i; ++j) {
                if (strcmp(hosts[i], hosts[j]) == 0) {
                    fprintf(stderr, "FAIL: ranks %d and %d report the same host\n", j, i);
                    ++errors;
                }
            }
        }
        fflush(stdout);
    }

    /* Every rank sends its rank to its successor and receives its predecessor. */
    int previous = (rank + size - 1) % size;
    int next = (rank + 1) % size;
    int received = -1;
    MPI_Sendrecv(&rank, 1, MPI_INT, next, 0,
                 &received, 1, MPI_INT, previous, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (received != previous) ++errors;

    int sum = -1;
    MPI_Allreduce(&rank, &sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (sum != 6) ++errors;
    int total_errors = 0;
    MPI_Allreduce(&errors, &total_errors, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) {
        if (total_errors == 0)
            printf("PASS: 4 ranks on 4 distinct hosts; ring exchange OK; Allreduce sum=6\n");
        else
            fprintf(stderr, "FAIL: %d check errors\n", total_errors);
    }
    MPI_Finalize();
    return total_errors ? 1 : 0;
}

