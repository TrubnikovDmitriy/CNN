#ifndef CONVOLUTIONAL_NEURAL_NETWORK_NEURON_HPP
#define CONVOLUTIONAL_NEURAL_NETWORK_NEURON_HPP

#include <vector>
#include "Utils.hpp"

class Neuron {
public:
    Neuron() {};
    virtual ~Neuron() {};
    virtual float getOut() = 0;

    void addOutputSynapce(Synapse* new_synapse);
    std::vector<Synapse*> getOutputSynapses();

private:
    std::vector<Synapse*> outputSynapces;
};



class BiasNeuron: public Neuron {
    // Псевдо-нейрон для смещения

    // Чтобы подключить его к нейрону, в конструкторе синапса,
    // BiasNeuron необходимо указать как входной,
    // а нейрон, к которому подключают, как выходной.
public:
    BiasNeuron() {};
    ~BiasNeuron() {};

    float getOut() { return 1.0f; };
};
class InputNeuron: public Neuron {
    // Не имеет входных синапсов,
    // позволяет устанавливать значение напрямую
public:
    InputNeuron(): input_data(0.0f) {};
    ~InputNeuron() {};

    void setInput(float data);
    float getOut();
private:
    float input_data;
};
class HiddenNeuron: public Neuron {
    // Имеет входные синапсы,
    // получает данные только через них.
public:
    HiddenNeuron() {};
    ~HiddenNeuron() {};

    void work();
    float getOut();
    void addInputSynapce(Synapse* new_synapse);
    std::vector<Synapse*> getInputSynapses();

private:
    std::vector<Synapse*> inputSynapces;
    float out;
};


#endif //CONVOLUTIONAL_NEURAL_NETWORK_NEURON_HPP
