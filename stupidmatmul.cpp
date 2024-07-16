#include <iostream>
#include <chrono>
#include <vector>
#include <cstdlib> // For srand() and rand()
#include <ctime>   // For time()
using namespace std;
using namespace std::chrono;
// Function to initialize a matrix with random float values
void initialize_matrix(float** matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = static_cast<float>(rand()) / RAND_MAX; // Random float between 0 and 1
        }
    }
}

// Function to allocate memory for a matrix
float** allocate_matrix(int rows, int cols) {
    float** matrix = new float*[rows];
    for (int i = 0; i < rows; ++i) {
        matrix[i] = new float[cols];
    }
    return matrix;
}

// Function to free allocated memory for a matrix
void free_matrix(float** matrix, int rows) {
    for (int i = 0; i < rows; ++i) {
        delete[] matrix[i];
    }
    delete[] matrix;
}

int main() {
    const int R1 = 1000, C1 = 1000;
    const int R2 = 1000, C2 = 1000;

    // Seed for random number generation
    srand(static_cast<unsigned>(time(0)));

    // Allocate memory for matrices
    float** mat1 = allocate_matrix(R1, C1);
    float** mat2 = allocate_matrix(R2, C2);
    float** rslt = allocate_matrix(R1, C2);

    // Initialize matrices with random values
    initialize_matrix(mat1, R1, C1);
    initialize_matrix(mat2, R2, C2);

    // Perform matrix multiplication
    auto start = system_clock().now();
    for (int i = 0; i < R1; ++i) {
        for (int j = 0; j < C2; ++j) {
            rslt[i][j] = 0;
            for (int k = 0; k < R2; ++k) {
                rslt[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
    }
    auto stop = system_clock().now();
    cout << "elapsed ms: " << duration_cast<milliseconds>(stop -start).count() << endl;
}
