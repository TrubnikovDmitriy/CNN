#ifndef CONVOLUTIONAL_NEURAL_NETWORK_MATRIX_HPP
#define CONVOLUTIONAL_NEURAL_NETWORK_MATRIX_HPP

#include <iostream>
#include <vector>

using std::vector;

template <class T>
class Matrix {
public:
    Matrix(u_int high, u_int width);
    Matrix(vector<vector<T>> _matrix);
    ~Matrix() {};

    Matrix<T> convolution(Matrix<T> kernel);
    void printMatrix();
private:
    vector<vector<T>> matrix;
};



// template realization
#include <cassert>

template <class T>
Matrix<T>::Matrix(u_int high, u_int width) {

    vector<T> temp(width, 0);
    for (int i = 0; i < high; ++i)
        matrix.push_back(temp);
}
template <class T>
Matrix<T>::Matrix(vector<vector<T>> _matrix): matrix(_matrix) {}

template <class T>
Matrix<T> Matrix<T>::convolution(Matrix kernel) {

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
template <class T>
void Matrix<T>::printMatrix() {

    for (auto row: matrix) {
        for (auto cell: row)
            printf("%5.2f ", (double)cell);
        std::cout << std::endl;
    }
    std::cout << std::endl;
}


#endif //CONVOLUTIONAL_NEURAL_NETWORK_MATRIX_HPP
