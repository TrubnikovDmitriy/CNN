#ifndef CONVOLUTIONAL_NEURAL_NETWORK_SYNAPSE_HPP
#define CONVOLUTIONAL_NEURAL_NETWORK_SYNAPSE_HPP

#include "Utils.hpp"

class Synapse {
public:
    Synapse(Neuron* _in, Neuron* _out);
    Synapse();
    ~Synapse() {};

    float getOutput();
    float getPrevDelta();
    void addWeight(float dw);
    float getWeight();
    Neuron* getOutputNeuron();
    Neuron* getInputNeuron();

private:
    float weight;
    float last_dw;
    Neuron* in;
    Neuron* out;
};

#endif //CONVOLUTIONAL_NEURAL_NETWORK_SYNAPSE_HPP
