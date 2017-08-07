#ifndef CONVOLUTIONAL_NEURAL_NETWORK_FILTER_HPP
#define CONVOLUTIONAL_NEURAL_NETWORK_FILTER_HPP

#include "Matrix.hpp"

class Matrix_3D {
public:
    // Конструкторы
    Matrix_3D(u_int high, u_int width, u_int depth, bool random = false);
    Matrix_3D(const vector<Matrix<float>>& matrixes, u_int depth);
    Matrix_3D(const vector<vector<float>>& matrix, u_int depth);
    Matrix_3D(const u_int depth);
    ~Matrix_3D() {};

    // Операторы
    float& operator()(const u_int h, const u_int w, const u_int d);
    Matrix<float>& operator[](const u_int depth);
    Matrix_3D operator+(const Matrix_3D& right);
    void operator=(const Matrix_3D& value);
    void operator+=(const Matrix_3D& value);

    // Свёртка
    Matrix<float> convolution(const Matrix_3D& kernel);

    // Служебные функции
    u_int getHigh();
    u_int getWidth();
    u_int getDepth();
    void printMatrix_3D();

private:
    unsigned int depth;
    vector<Matrix<float>> matrix_3D;
};


#endif //CONVOLUTIONAL_NEURAL_NETWORK_FILTER_HPP
