#include <iostream>
#include <vector>
#include "Layers.hpp"
#include "Matrix.hpp"
#include "Matrix_3D.hpp"

using namespace std;

int main() {

    srand((u_int)time(0));

    Matrix_3D kernel1(5, 4, 3, true);
    kernel1.printMatrix_3D();
    Matrix_3D kernel2(5, 4, 3, true);
    kernel2.printMatrix_3D();

    Matrix_3D kernel = kernel1 + kernel2;
    kernel.printMatrix_3D();
    printf("H = %d, W = %d, D = %d\n", kernel.getHigh(), kernel.getWidth(), kernel.getDepth());


    return 0;
}