#include <cassert>
#include <random>
#include "Synapse.hpp"
#include "Neuron.hpp"


Synapse::Synapse(Neuron *_in, Neuron *_out) {

    in = _in;
    out = _out;
    last_dw = 0;
    weight = random() % 10 - 5;
}
Synapse::Synapse() {

    in = nullptr;
    out = nullptr;
    last_dw = 0;
    weight = random() % 10 - 5;
}

float Synapse::getOutput() {
    // Дергаем синапсы - заставляем их получать
    // значения из предыдущего слоя нейронов.
    return weight * in->getOut();
}

float Synapse::getPrevDelta() {
    return last_dw;
}
void Synapse::addWeight(float dw) {

    last_dw = dw;
    weight += dw;
}

float Synapse::getWeight() {
    return weight;
}
Neuron* Synapse::getInputNeuron() {
    return in;
}
Neuron* Synapse::getOutputNeuron() {
    return out;
}