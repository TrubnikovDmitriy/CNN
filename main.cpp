#include <iostream>
#include <vector>
#include "Layers.hpp"


using namespace std;

int main() {

    srand((u_int)time(0));


    cout << "Input data" << endl;
    Matrix_3D input_data(7, 5, 3, true);
    input_data.printMatrix_3D();
    cout << endl << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;


    cout << "Два фильтра" << endl;
    ConvolutionalLayer convol(2, 2, 2, 3);
    convol.getOut();
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
    RectifierLayer ReLU(&convol);
    ReLU.getOut();
    ReLU.work();
    ReLU.getOut().printMatrix_3D();

    cout << endl << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;


    cout << "Три фильтра" << endl;
    ConvolutionalLayer convol2(3, 3, 3, &ReLU);
    convol2.getOut();
    vector<Matrix_3D> filters2 = convol2.getFilters();
    for (u_int i = 0; i < filters2.size(); ++i) {
        filters2[i].printMatrix_3D();
        cout << endl;
    }
    cout << endl << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;


    cout << "Output data" << endl;
    convol2.work();
    convol2.getOut().printMatrix_3D();
    cout << endl << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;

    cout << "Pool" << endl;
    PoolingLayer pool(&convol2, 2);
    pool.getOut();
    pool.work();
    pool.getOut().printMatrix_3D();
    cout << endl << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;


    cout << "Transfer" << endl;
    TransferLayer transfer(&pool);
    transfer.getOut();
    transfer.work();
    for (auto value: transfer.getOut())
        cout << value->getOut() << " ";
    cout << endl;
    cout << endl << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;


    cout << "Hidden #1" << endl;
    HiddenLayer hidden1(transfer.getOut(), 10);
    hidden1.getOut();
    hidden1.work();
    for (auto value: hidden1.getOut())
        printf("%5.2f ", value->getOut());
    cout << endl;
    cout << endl << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;

    cout << "Hidden #2" << endl;
    HiddenLayer hidden2(hidden1.getOut(), 3);
    hidden2.getOut();
    hidden2.work();
    for (auto value: hidden2.getOut())
        printf("%5.2f ", value->getOut());
    cout << endl;
    cout << endl << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;


    return 0;
}