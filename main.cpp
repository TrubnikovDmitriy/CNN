#include <iostream>
#include <vector>
#include "Matrix.hpp"

using namespace std;

int main() {

    vector<vector<float>> vec {
            {9, 3, 3},
            {1, 2, 3},
            {4, 5, -6}
    };
    vector<vector<float>> ker {
            {1, -1},
            {-1, 1}
    };
    Matrix<float> matrix(vec);
    Matrix<float> kernel(ker);
    Matrix<float> result = matrix.convolution(kernel);

    matrix.printMatrix();
    kernel.printMatrix();
    result.printMatrix();

    return 0;
}