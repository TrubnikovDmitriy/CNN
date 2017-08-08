#include <cassert>
#include <random>
#include "Synapse.hpp"
#include "Neuron.hpp"


Synapse::Synapse(Neuron *_in, HiddenNeuron *_out, float _weight): in(_in), out(_out) {

    // Связь двусторонняя (!) синапс с нейронами, нейроны с синапсами.
    // Синапс связывается со входным/выходным нейронами,
    // указанные нейроны добавляют синапс в свои вектора входных/выходных синапсов.
    // _weight - усстанавливает в текущее значение коэффициента синапса указанный вес.
    // В случае, если _weight = 0 (по умолчанию), значение веса выбирается рандомно.

    assert(in != nullptr);
    assert(out != nullptr);

    in->addOutputSynapce(this);
    out->addInputSynapce(this);

    last_dw = 0;
    if (_weight == 0.0f)
        weight = random() % 10 - 5;
    else
        weight = _weight;
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