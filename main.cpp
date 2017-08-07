#include <iostream>
#include <vector>
#include "Layers.hpp"
#include "Matrix.hpp"
#include "Matrix_3D.hpp"

using namespace std;

int main() {

    srand((u_int)time(0));

    cout << "Input data" << endl;
    Matrix_3D input_data(5, 4, 3, true);
    input_data.printMatrix_3D();
    cout << endl << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;


    cout << "Два фильтра" << endl;
    ConvolutionalLayer convol(2, 2, 2, 3);
    vector<Matrix_3D> filters = convol.getFilters();
    for (u_int i = 0; i < filters.size(); ++i) {
        filters[i].printMatrix_3D();
        cout << endl;
    }
    cout << endl << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;


    cout << "Output data" << endl;
    convol.work(input_data);
    convol.getOut().printMatrix_3D();
    cout << endl << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;


    return 0;
}