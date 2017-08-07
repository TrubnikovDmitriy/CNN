#include <iostream>
#include <vector>
#include "Layers.hpp"
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
    vector<vector<float>> ker2 {
            {-1, 1},
            {-1, 1}
    };
    vector<vector<float>> empty;


    Matrix<float> matrix(vec);
    Matrix<float> kernel(ker);
    Matrix<float> kernel2(ker2);
    Matrix<float> result = matrix.convolution(kernel);


    Matrix<float> sum = kernel + kernel2;
    kernel.printMatrix();

    kernel += kernel2;



//    matrix.printMatrix();
    kernel.printMatrix();
//    kernel2.printMatrix();
    sum.printMatrix();
//
//
//    result.printMatrix();
//
//    result(1, 0) = -10.2f;
//
//    result.printMatrix();
//

    return 0;
}