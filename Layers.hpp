#ifndef CONVOLUTIONAL_NEURAL_NETWORK_LAYERS_HPP
#define CONVOLUTIONAL_NEURAL_NETWORK_LAYERS_HPP

#include <iostream>
#include "Neuron.hpp"
#include "Matrix_3D.hpp"

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
    ConvolutionalLayer(u_int size, u_int width, u_int high, u_int depth);
    ~ConvolutionalLayer() {};

    void updateKernel(vector<Matrix_3D> deltaWeights, float moment);
    void work(Matrix_3D input_data);
    vector<Matrix_3D> getFilters();
    Matrix_3D getOut();

private:
    Matrix_3D feature_maps;
    vector<Matrix_3D> filters;
    vector<Matrix_3D> prevDeltaWeights;
};




#endif //CONVOLUTIONAL_NEURAL_NETWORK_LAYERS_HPP
