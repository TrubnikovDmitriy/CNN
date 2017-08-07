#include "Layers.hpp"
#include <algorithm>
#include <random>


// Layer
Layer::Layer(layers _type): type(_type) {

}
layers Layer::getType() {
    return type;
}

// HiddenLayer
HiddenLayer::HiddenLayer(u_int _size):
        Layer(layers::hidden), size(_size),
        neurons(size, new HiddenNeuron()) {}
void HiddenLayer::work() {
    for(auto neuron: neurons)
        neuron->calculate();
}
vector<float> HiddenLayer::getOut() {

    vector<float> out;
    for (auto neuron: neurons)
        out.push_back(neuron->getOut());

    return out;
}

// OutputLayer
OutputLayer::OutputLayer(u_int _size):
        Layer(layers::output), size(_size),
        neurons(size, new HiddenNeuron()) {}
void OutputLayer::work() {
    for(auto neuron: neurons)
        neuron->calculate();
}
vector<float> OutputLayer::getOut() {

    vector<float> out;
    for (auto neuron: neurons)
        out.push_back(neuron->getOut());

    return out;
}

// ConvolutionalLayer
ConvolutionalLayer::ConvolutionalLayer(u_int size, u_int width, u_int high, u_int depth):
                                                Layer(layers::convolutional), feature_maps(size) {

    for (u_int i = 0; i < size; ++i) {
        filters.push_back(Matrix_3D(width, high, depth, true));
        prevDeltaWeights.push_back(Matrix_3D(width, high, depth, false));
    }
}
void ConvolutionalLayer::work(Matrix_3D input_data) {

    // На вход подается 3х-мерное изображение (высота, ширина, глубина - ex.RGB)
    // Внутри свёрточного слоя имеется вектор трёхмерных фильтров (т.е. 4х-мерный свёрточный слой)
    // Глубина входного изображения и глубина каждого из фильтров ДОЛЖНЫ совпадать.
    // Каждый трехмерный фильтр сворачивается с трехмерный изображением,
    // в результате получается 2х-мерная матрица свойств (feature maps).
    // Выход определяется количеством фильтров и является вектором из feature maps (т.е. 3х-мерным)

    for (u_int i = 0; i < filters.size(); ++i)
        feature_maps[i] = input_data.convolution(filters[i]);

    // За двумя строчками скрыто дофигища операций
}

Matrix_3D ConvolutionalLayer::getOut() {

    assert(feature_maps.getHigh() != 0);
    assert(feature_maps.getWidth() != 0);
    assert(feature_maps.getDepth() != 0);

    return feature_maps;
}
vector<Matrix_3D>ConvolutionalLayer::getFilters() {
    return filters;
}
void ConvolutionalLayer::updateKernel(vector<Matrix_3D> deltaWeights, float moment) {

    // Для каждого фильтра обновляем веса
    assert(filters.size() == deltaWeights.size());

    for (size_t i = 0; i < deltaWeights.size(); ++i) {

        assert(filters[i].getHigh() == deltaWeights[i].getHigh());
        assert(filters[i].getWidth() == deltaWeights[i].getWidth());
        assert(filters[i].getDepth() == deltaWeights[i].getDepth());

        // На вход подаются направления градиента,
        // т.е. значения, на которые нужно уменьшить матрицу весов
        // (с учетом коэффициента момента инерции).

        for (u_int h = 0; h < filters[i].getHigh(); ++h)
            for (u_int w = 0; w < filters[i].getWidth(); ++w)
                for (u_int d = 0; d < filters[i].getDepth(); ++d)
                    filters[i](h, w, d) -= deltaWeights[i](h, w, d) + moment * prevDeltaWeights[i](h, w, d);

        prevDeltaWeights[i] = deltaWeights[i];
    }
}

// RectifierLayer
RectifierLayer::RectifierLayer(u_int size, float _ratio): Layer(layers::ReLU),
                                                                            ratioReLU(_ratio),
                                                                            output_data(size) {

    assert(size > 0);

    for (u_int i = 0; i < size; ++i)
        bias_neurons.push_back(1.0f);

}
void RectifierLayer::work(Matrix_3D input_data) {

    assert(bias_neurons.size() == input_data.getDepth());

    for (u_int d = 0; d < input_data.getDepth(); ++d) {
        output_data[d] = input_data[d];
        for (u_int w = 0; w < input_data.getWidth(); ++w) {
            for (u_int h = 0; h < input_data.getHigh(); ++h) {
                output_data(h, w, d) = rectifier(input_data(h, w, d) + bias_neurons[d]);
            }
        }
    }
}
Matrix_3D RectifierLayer::getOut() {
    return output_data;
}
float RectifierLayer::rectifier(float x) {

    return std::max(ratioReLU * x, x);
}


// PoolingLayer
PoolingLayer::PoolingLayer(u_int size, u_int _step): Layer(layers::pooling),
                                                     step(_step), output(size) {}
Matrix_3D PoolingLayer::getOut() {

    assert(output.getHigh() != 0);
    assert(output.getWidth() != 0);
    
    return output;
}
void PoolingLayer::work(Matrix_3D input) {
    
    assert(output.getDepth() == input.getDepth());
    
    u_int new_width = input.getWidth() / step;
    u_int new_high = input.getHigh() / step;
    
    if (input.getWidth() % step != 0) ++new_width;
    if (input.getHigh() % step != 0) ++new_high;

    Matrix_3D pooling_matrix(new_high, new_width, input.getDepth());

    for (u_int d = 0; d < input.getDepth(); ++d) {
        for (u_int w = 0; w < input.getWidth(); w += step) {
            for (u_int h = 0; h < input.getHigh(); h += step) {
                pooling_matrix(h / step, w / step, d) = getMax(input, h, w, d);
            }
        }
        output[d] = pooling_matrix[d];
    }
    
    
}
float PoolingLayer::getMax(Matrix_3D& input, u_int h, u_int w, u_int d) {

    float max_value = 0.0f;

    for (u_int i = h; i < h + step; ++i) {
        for (u_int j = w; j < w + step; ++j) {

            if (i < input.getHigh() && j < input.getWidth()) {
                if (max_value < input(i, j, d))
                    max_value = input(i, j, d);
            }
        }
    }

    return max_value;
}