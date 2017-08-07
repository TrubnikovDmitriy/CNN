#ifndef CONVOLUTIONAL_NEURAL_NETWORK_LAYERS_HPP
#define CONVOLUTIONAL_NEURAL_NETWORK_LAYERS_HPP

#include <iostream>
#include "Neuron.hpp"
#include "Matrix.hpp"

using std::vector;
enum layers {
    hidden = 1,
    output,
    convolutional,
    ReLU,
    pooling,
    average
};

class Layer {
public:
    Layer(layers _type);
    virtual ~Layer() {};

    layers getType();

protected:
    const layers type;
};

class HiddenLayer: protected Layer {
public:
    HiddenLayer(u_int _size);
    ~HiddenLayer() {};

    void work();
    vector<float> getOut();

private:
    const unsigned int size;
    vector<HiddenNeuron*> neurons;
};
class OutputLayer: protected Layer {
public:
    OutputLayer(u_int _size);
    ~OutputLayer() {};

    void work();
    vector<float> getOut();

private:
    const unsigned int size;
    vector<HiddenNeuron*> neurons;
};
class ConvolutionalLayer: protected Layer {
public:
    ConvolutionalLayer(u_int size, u_int width, u_int high);
    ~ConvolutionalLayer() {};

    void work(vector<Matrix<float>> input_data);
    vector<Matrix<float>> getOut();
    vector<Matrix<float>> getKernels();

    void updateKernel(vector<Matrix<float>> deltaWeights, float moment);

private:
    vector<Matrix<float>> convKernels;
    vector<Matrix<float>> prevDeltaWeights;
    vector<Matrix<float>> outMatrixes;
};




#endif //CONVOLUTIONAL_NEURAL_NETWORK_LAYERS_HPP
