#include <iostream>
#include <vector>
#include "Layers.hpp"
#include "Matrix.hpp"
#include "Matrix_3D.hpp"
#include "Neuron.hpp"

using namespace std;

int main() {

    srand((u_int)time(0));

    cout << "Input data" << endl;
    Matrix_3D input_data(7, 5, 3, true);
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


    cout << "ReLU" << endl;
    RectifierLayer ReLU(2, -0.0f);
    ReLU.setPrevLayer(&convol);
    ReLU.work();
    ReLU.getOut().printMatrix_3D();

    cout << endl << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;

    cout << "Pool" << endl;
    PoolingLayer pool(2, 2);
    pool.setPrevLayer(&ReLU);
    pool.work();
    pool.getOut().printMatrix_3D();
    cout << endl << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;


    cout << "Transfer" << endl;
    TransferLayer transfer(pool.getOut().getHigh(),
                           pool.getOut().getWidth(),
                           pool.getOut().getDepth());
    transfer.setPrevLayer(&pool);
    transfer.work();
    for (auto value: transfer.getOut())
        cout << value->getOut() << " ";
    cout << endl;
    cout << endl << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;


    cout << "Hidden #1" << endl;
    HiddenLayer hidden1(transfer.getOut(), 10);
    hidden1.work();
    for (auto value: hidden1.getOut())
        printf("%5.2f ", value->getOut());
    cout << endl;
    cout << endl << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;

    cout << "Hidden #2" << endl;
    HiddenLayer hidden2(hidden1.getOut(), 3);
    hidden2.work();
    for (auto value: hidden2.getOut())
        printf("%5.2f ", value->getOut());
    cout << endl;
    cout << endl << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;

    return 0;
}