#ifndef CONVOLUTIONAL_NEURAL_NETWORK_LAYERS_HPP
#define CONVOLUTIONAL_NEURAL_NETWORK_LAYERS_HPP

#include <iostream>
#include "Neuron.hpp"
#include "Matrix_3D.hpp"

using std::vector;
enum layers {
    hidden = 0,
    convolutional,
    ReLU,
    pooling,
    average,
    transfer
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
    HiddenLayer(vector<Neuron*> input_neurons, u_int output_size);
    ~HiddenLayer();

    void work();
    vector<Neuron*> getOut();

private:
    vector<HiddenNeuron*> output_neurons;
    vector<BiasNeuron*> bias_neurons;
    vector<Neuron*> output_data;
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
class RectifierLayer: protected Layer {
public:
    RectifierLayer(u_int size, float ratio = -0.0f);
    ~RectifierLayer() {};

    void work(Matrix_3D input_data);
    Matrix_3D getOut();

private:
    float rectifier(float x);
    const float ratioReLU;

    vector<float> bias_neurons;
    Matrix_3D output_data;
};
class PoolingLayer: protected Layer {
public:
    PoolingLayer(u_int size, u_int step);
    ~PoolingLayer() {};

    void work(Matrix_3D input);
    Matrix_3D getOut();

private:
    float getMax(Matrix_3D& input, u_int h, u_int w, u_int d);
    const unsigned int step;
    Matrix_3D output;
};
class TransferLayer: protected Layer {
public:
    TransferLayer(u_int high, u_int width, u_int depth);
    ~TransferLayer();

    u_int getNeuronPosition(u_int h, u_int w, u_int d);
    vector<Neuron*> getOut();
    void work(Matrix_3D input);
    u_int getSize();

private:
    vector<InputNeuron*> neurons;
    vector<Neuron*> output_neurons;
    const unsigned int high;
    const unsigned int width;
    const unsigned int depth;
};




#endif //CONVOLUTIONAL_NEURAL_NETWORK_LAYERS_HPP
