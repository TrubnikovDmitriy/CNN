#include "Layers.hpp"
#include "Synapse.hpp"
#include <algorithm>
#include <random>


// HiddenLayer
HiddenLayer::HiddenLayer(vector<Neuron*> input_neurons, u_int out_size) {

    // На вход подается предыдущий слой, текущий и предыдущие
    // слои жестко связываются с помощью синапсов.

    // Создаются только ВХОДНЫЕ синапсы для текущего слоя!
    // В деструкторе удаляются текущий слой нейронов
    // и слой входных для него синапсов.

    assert(input_neurons.size() != 0);


    for (u_int i = 0; i < out_size; ++i) {

        // Создание выходного нейрона
        output_neurons.push_back(new HiddenNeuron());

        // Связывание с помощью синапса двух нейроновов (входного и выходного).
        for (auto input_neuron: input_neurons) {
            new Synapse(input_neuron, output_neurons[i]);
        }

        // Навешиваем, дополнительно, нейрон-смещения на каждый выходной нейрон.
        bias_neurons.push_back(new BiasNeuron());
        new Synapse(bias_neurons[i], output_neurons[i], 1.0f);

        // Костыль, ибо не получается привести класс
        // к родительскому внутри вектора (см. TransferLayer)
        Neuron* temp = output_neurons[i];
        output_data.push_back(temp);
    }
}
HiddenLayer::~HiddenLayer() {

    for (auto output_neuron: output_neurons) {
        for (auto synapse: output_neuron->getInputSynapses())
            delete synapse;
        delete output_neuron;
    }
    for (auto bias_neuron: bias_neurons)
        delete bias_neuron;
}
void HiddenLayer::work() {
    for (auto neuron: output_neurons)
        neuron->work();
}
vector<Neuron*> HiddenLayer::getOut() {
    return output_data;
}

// ConvolutionalLayer
ConvolutionalLayer::ConvolutionalLayer(u_int filter_number,
                                       u_int filter_high,
                                       u_int filter_width,
                                       u_int filter_depth): feature_maps(filter_number) {

    for (u_int i = 0; i < filter_number; ++i) {
        filters.push_back(Matrix_3D(filter_high,
                                    filter_width,
                                    filter_depth, true));

        prevDeltaWeights.push_back(Matrix_3D(filter_high,
                                             filter_width,
                                             filter_depth,
                                             false));
    }
}

ConvolutionalLayer::ConvolutionalLayer(u_int filter_number,
                                       u_int filter_high,
                                       u_int filter_width,
                                       OutputMatrixLayer *prev_layer): feature_maps(filter_number) {

    setPrevLayer(prev_layer);

    for (u_int i = 0; i < filter_number; ++i) {

        filters.push_back(Matrix_3D(filter_high, filter_width,
                                    prevLayer->getOut().getDepth(), true));

        prevDeltaWeights.push_back(Matrix_3D(filter_high, filter_width,
                                             prevLayer->getOut().getDepth(), false));
    }
}

void ConvolutionalLayer::work(Matrix_3D input_data) {

    // На вход подается 3х-мерное изображение (высота, ширина, глубина - ex.RGB)
    // Внутри свёрточного слоя имеется вектор трёхмерных фильтров (т.е. 4х-мерный свёрточный слой)
    // Глубина входного изображения и глубина каждого из фильтров ДОЛЖНЫ совпадать.
    // Каждый трехмерный фильтр сворачивается с трехмерный изображением,
    // в результате получается 2х-мерная матрица свойств (feature maps).
    // Выход определяется количеством фильтров и является вектором из feature maps (т.е. 3х-мерным)

    for (u_int i = 0; i < filters.size(); ++i)
        feature_maps[i] = input_data.convolution(filters[i]);

    // За двумя строчками скрыто дофигища операций
}

Matrix_3D ConvolutionalLayer::getOut() {

    return feature_maps;
}
vector<Matrix_3D>ConvolutionalLayer::getFilters() {
    return filters;
}
void ConvolutionalLayer::updateKernel(vector<Matrix_3D> deltaWeights, float moment) {

    // Для каждого фильтра обновляем веса
    assert(filters.size() == deltaWeights.size());

    for (size_t i = 0; i < deltaWeights.size(); ++i) {

        assert(filters[i].getHigh() == deltaWeights[i].getHigh());
        assert(filters[i].getWidth() == deltaWeights[i].getWidth());
        assert(filters[i].getDepth() == deltaWeights[i].getDepth());

        // На вход подаются направления градиента,
        // т.е. значения, на которые нужно уменьшить матрицу весов
        // (с учетом коэффициента момента инерции).

        for (u_int h = 0; h < filters[i].getHigh(); ++h)
            for (u_int w = 0; w < filters[i].getWidth(); ++w)
                for (u_int d = 0; d < filters[i].getDepth(); ++d)
                    filters[i](h, w, d) -= deltaWeights[i](h, w, d) + moment * prevDeltaWeights[i](h, w, d);

        prevDeltaWeights[i] = deltaWeights[i];
    }
}
void ConvolutionalLayer::work() {

    assert(prevLayer != nullptr);
    work(prevLayer->getOut());
}

// RectifierLayer
RectifierLayer::RectifierLayer(u_int depth, float _ratio): ratioReLU(_ratio),
                                                           output_data(depth) {

    assert(depth > 0);

    for (u_int i = 0; i < depth; ++i)
        bias_neurons.push_back(1.0f);

}
RectifierLayer::RectifierLayer(OutputMatrixLayer *prev_layer, float _ratio): ratioReLU(_ratio),
                                                                             output_data(prev_layer->getOut().getDepth()) {

    setPrevLayer(prev_layer);

    for (u_int i = 0; i < output_data.getDepth(); ++i)
        bias_neurons.push_back(1.0f);
}
void RectifierLayer::work(Matrix_3D input_data) {

    assert(bias_neurons.size() == input_data.getDepth());

    for (u_int d = 0; d < input_data.getDepth(); ++d) {
        output_data[d] = input_data[d];
        for (u_int w = 0; w < input_data.getWidth(); ++w) {
            for (u_int h = 0; h < input_data.getHigh(); ++h) {
                output_data(h, w, d) = rectifier(input_data(h, w, d) + bias_neurons[d]);
            }
        }
    }
}
Matrix_3D RectifierLayer::getOut() {
    return output_data;
}
float RectifierLayer::rectifier(float x) {

    return std::max(ratioReLU * x, x);
}
void RectifierLayer::work() {

    assert(prevLayer != nullptr);
    work(prevLayer->getOut());
}

// PoolingLayer
PoolingLayer::PoolingLayer(u_int size, u_int _step): step(_step), output(size) {}
PoolingLayer::PoolingLayer(OutputMatrixLayer *prev_layer, u_int _step): step(_step),
                                                                        output(prev_layer->getOut().getDepth()) {
    setPrevLayer(prev_layer);
}
Matrix_3D PoolingLayer::getOut() {

    return output;
}
void PoolingLayer::work(Matrix_3D input) {
    
    assert(output.getDepth() == input.getDepth());
    
    u_int new_width = input.getWidth() / step;
    u_int new_high = input.getHigh() / step;
    
    if (input.getWidth() % step != 0) ++new_width;
    if (input.getHigh() % step != 0) ++new_high;

    Matrix_3D pooling_matrix(new_high, new_width, input.getDepth());

    for (u_int d = 0; d < input.getDepth(); ++d) {
        for (u_int w = 0; w < input.getWidth(); w += step) {
            for (u_int h = 0; h < input.getHigh(); h += step) {
                pooling_matrix(h / step, w / step, d) = getMax(input, h, w, d);
            }
        }
        output[d] = pooling_matrix[d];
    }
    
    
}
float PoolingLayer::getMax(Matrix_3D& input, u_int h, u_int w, u_int d) {

    float max_value = 0.0f;

    for (u_int i = h; i < h + step; ++i) {
        for (u_int j = w; j < w + step; ++j) {

            if (i < input.getHigh() && j < input.getWidth()) {
                if (max_value < input(i, j, d))
                    max_value = input(i, j, d);
            }
        }
    }

    return max_value;
}
void PoolingLayer::work() {

    assert(prevLayer != nullptr);
    work(prevLayer->getOut());
}

// TransferLayer
TransferLayer::TransferLayer(u_int h, u_int w, u_int d): high(h), width(w), depth(d) {

    // Переходной слой - переход от трехмерной матрицы к нейронам,
    // при создании необходимо указать размер входной 3х-мерной матрицы.
    // На вход подается матрица, на выходе - вектор нейронов.

    assert(high && width && depth);

    for (u_int i = 0; i < depth; ++i) {
        for (u_int j = 0; j < width; ++j) {
            for (u_int k = 0; k < high; ++k) {
                neurons.push_back(new InputNeuron());

                // Довольно странный костыль
                // vector<DerivativeClass> никоим образом не хочет кастоваться к vector<BaseClass>,
                // несмотря на то, что последний является родителем первого.

                // Поэтому решено хранить два вектора с одинаковыми адресами
                // указателей, но разными типами указателей.
                Neuron* temp = neurons.back();
                output_neurons.push_back(temp);
            }
        }
    }

}

TransferLayer::TransferLayer(OutputMatrixLayer *prev_layer): high(prev_layer->getOut().getHigh()),
                                                             width(prev_layer->getOut().getWidth()),
                                                             depth(prev_layer->getOut().getDepth()) {

    setPrevLayer(prev_layer);
    // Комментарии смотри в предыдущем конструкторе

    assert(high && width && depth);

    for (u_int i = 0; i < depth; ++i) {
        for (u_int j = 0; j < width; ++j) {
            for (u_int k = 0; k < high; ++k) {
                neurons.push_back(new InputNeuron());

                Neuron* temp = neurons.back();
                output_neurons.push_back(temp);
            }
        }
    }
}
TransferLayer::~TransferLayer() {
    for (auto neuron: neurons)
        delete neuron;
}
void TransferLayer::work(Matrix_3D input) {

    assert(depth == input.getDepth());
    assert(width == input.getWidth());
    assert(high == input.getHigh());

    int k = 0;
    for (u_int d = 0; d < depth; ++d) {
        for (u_int w = 0; w < width; ++w) {
            for (u_int h = 0; h < high; ++h) {
                neurons[k++]->setInput(input(h, w, d));
            }
        }
    }
}
u_int TransferLayer::getNeuronPosition(u_int h, u_int w, u_int d) {
    return (h + high * w + high * width * d);
}
vector<Neuron*> TransferLayer::getOut() {
    return output_neurons;
}
u_int TransferLayer::getSize() {
    return high * width * depth;
}
void TransferLayer::work() {

    assert(prevLayer != nullptr);
    work(prevLayer->getOut());
}