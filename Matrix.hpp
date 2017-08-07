#ifndef CONVOLUTIONAL_NEURAL_NETWORK_MATRIX_HPP
#define CONVOLUTIONAL_NEURAL_NETWORK_MATRIX_HPP

#include <iostream>
#include <vector>

using std::vector;

template <class T>
class Matrix {
public:
    Matrix(): high(0), width(0) {};
    Matrix(u_int high, u_int width, bool random = false);
    Matrix(vector<vector<T>> _matrix);
    ~Matrix() {};

    // Операторы
    T& operator()(const u_int w, const u_int h);
    Matrix<T> operator+(const Matrix<T>& right);
    void operator+=(const Matrix<T>& value);
    void operator=(const Matrix<T>& value);

    // Свёртка
    Matrix<T> convolution(const Matrix<T>& kernel);

    // Служебные функции
    u_int getHigh();
    u_int getWidth();
    void printMatrix();

private:
    vector<vector<T>> matrix;
    unsigned int high;
    unsigned int width;
};



// template realization
#include <cassert>
#include <ctime>
#include <random>

// Конструкторы
template <class T>
Matrix<T>::Matrix(u_int _high, u_int _width, bool random): high(_high),
                                                           width(_width) {

    assert(high > 0);
    assert(width > 0);

    vector<T> temp(width, 0);

    for (u_int i = 0; i < high; ++i) {
        if (random) {
            for (u_int j = 0; j < width; ++j)
                temp[j] = (T)((rand() % 20) - 10);
        }
        matrix.push_back(temp);
    }
}
template <class T>
Matrix<T>::Matrix(vector<vector<T>> _matrix): matrix(_matrix) {

    assert(matrix.size() != 0);
    high = (u_int)matrix.size();
    width = (u_int)matrix[0].size();
}


// Свёртка
template <class T>
Matrix<T> Matrix<T>::convolution(const Matrix<T>& kernel) {

    assert(kernel.matrix.size() > 0);

    // Проверяем, что ядро свертки меньше сворачиваемой
    // матрицы, заодно получаем размер результирующей
    int new_width = (int)(matrix.size() - kernel.matrix.size() + 1);
    int new_high = (int)(matrix[0].size() - kernel.matrix[0].size() + 1);
    assert(new_width > 0);
    assert(new_high > 0);

    vector<vector<T>> result;
    result.resize((size_t)new_high);


    // Поэлементное умножение (свёртка) матриц
    float sum = 0;
    float slag = 0;
    for (int i = 0; i < new_high; ++i) {
        for (int j = 0; j < new_width; ++j) {
            sum = 0;
            for (u_int ii = 0; ii < kernel.matrix.size(); ++ii) {
                for (u_int jj = 0; jj < kernel.matrix[0].size(); ++jj) {
                    slag = matrix[i + ii][j + jj] * kernel.matrix[ii][jj];
                    sum += slag;
                }
            }
            result[i].push_back(sum);
        }
    }

    Matrix<T> result_matrix(result);
    return result_matrix;
}


// Операторы
template <class T>
T& Matrix<T>::operator()(const u_int h, const u_int w) {

    assert(h < matrix.size());
    assert(w < matrix[h].size());
    return matrix[h][w];
}
template <class T>
Matrix<T> Matrix<T>::operator+(const Matrix<T>& right) {

    assert(high == right.high);
    assert(width == right.width);

    Matrix<T> sum(high, width);

    for (u_int h = 0; h < high; ++h) {
        for (u_int w = 0; w < width; ++w) {
            sum.matrix[h][w] = matrix[h][w] + right.matrix[h][w];
        }
    }
    return sum;
}
template <class T>
void Matrix<T>::operator=(const Matrix<T> &value) {

    high = value.high;
    width = value.width;
    matrix = value.matrix;
}
template <class T>
void Matrix<T>::operator+=(const Matrix<T> &value) {
    *this = *this + value;
}


// Служебные функции
template <class T>
void Matrix<T>::printMatrix() {

    for (auto row: matrix) {
        for (auto cell: row)
            printf("%7.2f ", (double)cell);
        std::cout << std::endl;
    }
    std::cout << std::endl;
}
template <class T>
u_int Matrix<T>::getHigh() { return high; }
template <class T>
u_int Matrix<T>::getWidth() { return width; }


#endif //CONVOLUTIONAL_NEURAL_NETWORK_MATRIX_HPP
