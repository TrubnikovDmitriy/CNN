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
    virtual void work() = 0;

    layers getType();

protected:
    const layers type;
};

class InputMatrixLayer;
class InputNeuronLayer;
class OutputMatrixLayer;
class OutputNeuronLayer;

class InputMatrixLayer:     protected virtual Layer {
public:
    InputMatrixLayer(layers type): Layer(type), prevLayer(nullptr) {};
    virtual void work(Matrix_3D input) = 0;
    void setPrevLayer(OutputMatrixLayer* prev) {
        prevLayer = prev;
    };
    virtual void work() = 0;

protected:
    OutputMatrixLayer* prevLayer;
};
class InputNeuronLayer:     protected virtual Layer {
public:
    InputNeuronLayer(layers type): Layer(type), prevLayer(nullptr) {};
    void setPrevLayer(OutputNeuronLayer* prev) {
        prevLayer = prev;
    };
    virtual void work() = 0;

protected:
    OutputNeuronLayer* prevLayer;
};
class OutputMatrixLayer:    protected virtual Layer {
public:
    OutputMatrixLayer(layers type): Layer(type), nextLayer(nullptr) {};
    void setNextLayer(InputMatrixLayer* next) {
        nextLayer = next;
    };
    virtual Matrix_3D getOut() = 0;
    virtual void work() = 0;

protected:
    InputMatrixLayer* nextLayer;
};
class OutputNeuronLayer:    protected virtual Layer {
public:
    OutputNeuronLayer(layers type): Layer(type), nextLayer(nullptr) {};
    virtual vector<Neuron*> getOut() = 0;
    void setNextLayer(InputNeuronLayer* next) {
        nextLayer = next;
    };
    virtual void work() = 0;

protected:
    InputNeuronLayer* nextLayer;
};


class ConvolutionalLayer:   public InputMatrixLayer, public OutputMatrixLayer {
public:
    ConvolutionalLayer(u_int filter_number, u_int width, u_int high, u_int depth);
    ConvolutionalLayer(u_int filter_number, OutputMatrixLayer* prev_layer);
    ~ConvolutionalLayer() {};

    void updateKernel(vector<Matrix_3D> deltaWeights, float moment);
    void work(Matrix_3D input_data);
    void work();
    vector<Matrix_3D> getFilters();
    Matrix_3D getOut();

private:
    Matrix_3D feature_maps;
    vector<Matrix_3D> filters;
    vector<Matrix_3D> prevDeltaWeights;
};
class RectifierLayer:       public InputMatrixLayer, public OutputMatrixLayer {
public:
    RectifierLayer(OutputMatrixLayer* prev_layer, float ratio = -0.0f);
    RectifierLayer(u_int depth, float ratio = -0.0f);
    ~RectifierLayer() {};

    void work(Matrix_3D input_data);
    void work();
    Matrix_3D getOut();

private:
    float rectifier(float x);
    const float ratioReLU;

    vector<float> bias_neurons;
    Matrix_3D output_data;
};
class TransferLayer:        public InputMatrixLayer, public OutputNeuronLayer {
public:
    TransferLayer(u_int high, u_int width, u_int depth);
    TransferLayer(OutputMatrixLayer* prev_layer);
    ~TransferLayer();

    u_int getNeuronPosition(u_int h, u_int w, u_int d);
    vector<Neuron*> getOut();
    void work(Matrix_3D input);
    void work();
    u_int getSize();

private:
    vector<InputNeuron*> neurons;
    vector<Neuron*> output_neurons;
    const unsigned int high;
    const unsigned int width;
    const unsigned int depth;
};
class PoolingLayer:         public InputMatrixLayer, public OutputMatrixLayer {
public:
    PoolingLayer(OutputMatrixLayer* prev_layer, u_int step);
    PoolingLayer(u_int size, u_int step);
    ~PoolingLayer() {};

    void work(Matrix_3D input);
    void work();
    Matrix_3D getOut();

private:
    float getMax(Matrix_3D& input, u_int h, u_int w, u_int d);
    const unsigned int step;
    Matrix_3D output;
};
class HiddenLayer:          public InputNeuronLayer, public OutputNeuronLayer {
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




#endif //CONVOLUTIONAL_NEURAL_NETWORK_LAYERS_HPP
