#include "Layers.hpp"
#include <ctime>
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
                                                Layer(layers::convolutional), feature_maps(depth) {

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