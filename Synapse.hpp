#ifndef CONVOLUTIONAL_NEURAL_NETWORK_SYNAPSE_HPP
#define CONVOLUTIONAL_NEURAL_NETWORK_SYNAPSE_HPP

#include "Utils.hpp"

class Synapse {
public:
    Synapse(Neuron* _in, HiddenNeuron* _out, float weight = 0.0f);
    Synapse();
    ~Synapse() {};

    float getOutput();
    float getPrevDelta();
    void addWeight(float dw);
    float getWeight();
    Neuron* getOutputNeuron();
    Neuron* getInputNeuron();

private:
    Neuron* in;
    HiddenNeuron* out;
    float weight;
    float last_dw;
};

#endif //CONVOLUTIONAL_NEURAL_NETWORK_SYNAPSE_HPP
