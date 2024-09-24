## How It Works
1. The matrix `A` is broadcasted to all processes.
2. The matrix `B` is scattered in columns among all processes.
3. Each process performs matrix multiplication on its assigned section of `B`.
4. The process with the largest element in local matrix `B` gathers and prints the final result.
