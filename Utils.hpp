#ifndef CONVOLUTIONAL_NEURAL_NETWORK_UTILS_HPP
#define CONVOLUTIONAL_NEURAL_NETWORK_UTILS_HPP

#include <cmath>

class Synapse;
class Neuron;
class HiddenNeuron;

inline float activateFunction(float x) {

    return (float)(1 / (1 + exp(-x)));
}
inline float diffActivate(float out) {

    return (1 - out) * out;
}

#endif //CONVOLUTIONAL_NEURAL_NETWORK_UTILS_HPP
