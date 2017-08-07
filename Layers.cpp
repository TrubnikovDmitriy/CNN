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
ConvolutionalLayer::ConvolutionalLayer(u_int size, u_int width, u_int high):
                                                Layer(layers::convolutional) {

    for (u_int i = 0; i < size; ++i) {
        convKernels.push_back(Matrix<float>(width, high, true));
        prevDeltaWeights.push_back(Matrix<float>(width, high));
    }
}
void ConvolutionalLayer::work(vector<Matrix<float>> input_data) {

    assert(input_data.size() != 0);
    outMatrixes.clear();

    // Каждон входное изображение нужно прогнать через ядро свертки,
    // а затем сложить результаты. И так для каждой свертки
    // (предполагается, что размерности входных данных одинаковы).

    for (auto convKernel: convKernels) {
        // Создаем нулевую матрицу для суммирования
        Matrix<float> sum(input_data[0].getHigh() - convKernel.getHigh() + 1,
                          input_data[0].getWidth() - convKernel.getWidth() + 1);

        // Складываем результаты свертки входных данных с данным фильтром
        for (auto input_matrix: input_data)
            sum += input_matrix.convolution(convKernel);

        // Добавляем полученную матрицу в выходной вектор (кол-во выходов = кол-ву фильтров)
        outMatrixes.push_back(sum);
    }
}

vector<Matrix<float>> ConvolutionalLayer::getOut() {
    return outMatrixes;
}
vector<Matrix<float>> ConvolutionalLayer::getKernels() {
    return convKernels;
}
void ConvolutionalLayer::updateKernel(vector<Matrix<float>> deltaWeights, float moment) {

    // Для каждого фильтра обновляем веса
    assert(convKernels.size() == deltaWeights.size());

    for (size_t i = 0; i < deltaWeights.size(); ++i) {

        assert(convKernels[i].getWidth() == deltaWeights[i].getWidth());
        assert(convKernels[i].getHigh() == deltaWeights[i].getHigh());

        // На вход подаются направления градиента,
        // т.е. значения, на которые нужно уменьшить матрицу весов
        // (с учетом коэффициента момента инерции).

        for (u_int h = 0; h < convKernels[i].getHigh(); ++h)
            for (u_int w = 0; w < convKernels[i].getWidth(); ++w)
                convKernels[i](h, w) -= deltaWeights[i](h, w) + moment * prevDeltaWeights[i](h, w);

        prevDeltaWeights[i] = deltaWeights[i];
    }
}