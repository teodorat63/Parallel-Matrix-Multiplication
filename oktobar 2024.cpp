#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>
#include <mpi.h>

#define N 3
#define R 3
#define K 6

int main(int argc, char** argv)
{
    int p, rank;
    int A[N][R], B[R][K], C[N][K] = { 0 };
    MPI_Datatype columns_not_resized, columns;
    MPI_Datatype new_columns_not_resized, new_columns;
    int local_max_val;
    int max_process_index;

    // Structure to hold local and global max values and process index
    struct {
        int max;
        int ind;
    } global_max, local_max;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    // Define column-based datatypes for scattering matrices
    MPI_Type_vector(K / p * R, 1, p, MPI_INT, &columns_not_resized);
    MPI_Type_commit(&columns_not_resized);
    MPI_Type_create_resized(columns_not_resized, 0, sizeof(int), &columns);
    MPI_Type_commit(&columns);

    MPI_Type_vector(N * K / p, 1, p, MPI_INT, &new_columns_not_resized);
    MPI_Type_commit(&new_columns_not_resized);
    MPI_Type_create_resized(new_columns_not_resized, 0, sizeof(int), &new_columns);
    MPI_Type_commit(&new_columns);

    // Allocate memory for local parts of matrices B and C
    int* local_B = (int*)malloc(R * (K / p) * sizeof(int));
    int* local_C = (int*)malloc(N * (K / p) * sizeof(int));

    srand(time(0)); // Initialize random seed

    // Initialize matrices A and B on process 0
    if (rank == 0) {
        for (int i = 0; i < R; i++) {
            for (int j = 0; j < N; j++) {
                A[j][i] = rand() % 10;
            }
            for (int l = 0; l < K; l++) {
                B[i][l] = rand() % 100;
            }
        }
    }

    // Broadcast matrix A to all processes
    MPI_Bcast(&A[0][0], N * R, MPI_INT, 0, MPI_COMM_WORLD);
    // Scatter matrix B's columns to all processes
    MPI_Scatter(B, 1, columns, local_B, R * K / p, MPI_INT, 0, MPI_COMM_WORLD);

    // Matrix multiplication: A * local_B
    for (int i = 0; i < N; i++) {
        for (int l = 0; l < K / p; l++) {
            local_C[i * (K / p) + l] = 0;
            for (int j = 0; j < R; j++) {
                local_C[i * (K / p) + l] += A[i][j] * local_B[j * (K / p) + l];
            }
        }
    }

    // Print result of local multiplication
    printf("%d: Result of multiplication:\n", rank);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K / p; j++) {
            printf("%d ", local_C[i * (K / p) + j]);
        }
        printf("\n");
    }

    // Find local maximum in the part of matrix B
    local_max_val = INT_MIN;
    for (int i = 0; i < R * K / p; i++) {
        if (local_B[i] > local_max_val) {
            local_max_val = local_B[i];
        }
    }

    // Set local max structure and find global max using MPI_Reduce
    local_max.ind = rank;
    local_max.max = local_max_val;

    MPI_Reduce(&local_max, &global_max, 1, MPI_2INT, MPI_MAXLOC, 0, MPI_COMM_WORLD);

    // Print the global maximum and process that holds it
    if (rank == 0) {
        max_process_index = global_max.ind;
        printf("The largest element is %d and it is located in process %d\n", global_max.max, global_max.ind);
    }

    // Broadcast the process that has the maximum value
    MPI_Bcast(&max_process_index, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Gather matrix C into the process with the maximum value
    MPI_Gather(local_C, N * K / p, MPI_INT, C, 1, new_columns, max_process_index, MPI_COMM_WORLD);

    // Process with the global maximum prints the result matrix C
    if (rank == max_process_index) {
        printf("%d: Matrix C (final result):\n", rank);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < K; j++) {
                printf("%d ", C[i][j]);
            }
            printf("\n");
        }
    }

    // Finalize MPI
    MPI_Finalize();

    // Free dynamically allocated memory
    free(local_B);
    free(local_C);

    return 0;
}
