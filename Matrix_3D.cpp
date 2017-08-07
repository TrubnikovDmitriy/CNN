#include "Matrix_3D.hpp"


// Конструкторы
Matrix_3D::Matrix_3D(u_int _high, u_int _width, u_int _depth, bool random):
                                                        depth(_depth) {

    // Упаковываем матрицы в вектор. В итоге, отдельный фильтр имеет 3 измерения,
    // где глубина должна соответствовать глубине входящих данных

    assert(depth != 0);

    for (u_int i = 0; i < depth; ++i)
        matrix_3D.push_back(Matrix<float>(_high, _width, random));

}
Matrix_3D::Matrix_3D(const vector<vector<float>>& matrix, u_int _depth):
                                                        depth(_depth) {

    assert(depth != 0);
    assert(matrix.size() != 0);
    assert(matrix[0].size() != 0);

    for (u_int i = 0; i < depth; ++i)
        matrix_3D.push_back(Matrix<float>(matrix));

}
Matrix_3D::Matrix_3D(const vector<Matrix<float>>& matrixes, u_int _depth):
                                                        depth(_depth) {
    assert(depth > 0);

    for (auto matrix: matrixes)
        matrix_3D.push_back(matrix);

}
Matrix_3D::Matrix_3D(const u_int _depth): depth(_depth) {

    assert(depth > 0);
    matrix_3D.resize(depth);
}

// Операторы
float& Matrix_3D::operator()(const u_int h, const u_int w, const u_int d) {

    assert(d < depth);
    return matrix_3D[d](h, w);
}
void Matrix_3D::operator=(const Matrix_3D &value) {

    assert(value.depth > 0);

    matrix_3D.clear();
    for (auto matrix: value.matrix_3D)
        matrix_3D.push_back(matrix);

    depth = value.depth;

}
Matrix_3D Matrix_3D::operator+(const Matrix_3D &right) {

    assert(depth == right.depth);

    vector<Matrix<float>> sum_filter(depth);
    for (u_int i = 0; i < depth; ++i)
        sum_filter[i] = matrix_3D[i] + right.matrix_3D[i];

    return Matrix_3D(sum_filter, depth);
}
void Matrix_3D::operator+=(const Matrix_3D &value) {

    assert(depth == value.depth);
    for (u_int i = 0; i < depth; ++i)
        matrix_3D[i] += value.matrix_3D[i];
}
Matrix<float>& Matrix_3D::operator[](const u_int _depth) {

    assert(_depth < depth);
    return matrix_3D[_depth];
}

// Свертка
Matrix<float> Matrix_3D::convolution(const Matrix_3D &kernel) {

    assert(depth == kernel.depth);

    Matrix<float> sum = matrix_3D[0].convolution(kernel.matrix_3D[0]);

    for (u_int i = 1; i < depth; ++i)
        sum += matrix_3D[i].convolution(kernel.matrix_3D[i]);

    return sum;
}

// Служебные функции
u_int Matrix_3D::getHigh() { return matrix_3D[0].getHigh(); }
u_int Matrix_3D::getWidth() { return matrix_3D[0].getWidth(); }
u_int Matrix_3D::getDepth() { return depth; }

void Matrix_3D::printMatrix_3D() {

    for (u_int h = 0; h < matrix_3D[0].getHigh(); ++h) {
        for (u_int d = 0; d < depth; ++d) {
            for (u_int w = 0; w < matrix_3D[d].getWidth(); ++w)
                printf("%7.2f", matrix_3D[d](h, w));
            printf("\t\t");
        }
        printf("\n");
    }
    printf("\n");
}
