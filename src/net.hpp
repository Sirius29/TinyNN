#ifndef TINYNN_NET_H_
#define TINYNN_NET_H_

#include "layer.hpp"
#include <iostream>

namespace tinynn
{
    class Net
    {
    public:
        Net(std::vector<Layer *> _layers) : layers(_layers) {}
        ~Net() {}
        Tensor Forward(const Tensor &input)
        {
            Tensor out(input);
            for (Layer *layer : layers)
            {
                out = layer->Forward(out);
            }
            return out;
        }

        Tensor Backward(const Tensor &grad)
        {
            Tensor out(grad);
            for (int i = layers.size() - 1; i >= 0; --i)
            {
                out = layers[i]->Backward(out);
            }
            return out;
        }

        void Print()
        {
            std::cout << "Net Structure:" << std::endl;
        }

    private:
        std::vector<Layer *> layers;
    };
}

#endif