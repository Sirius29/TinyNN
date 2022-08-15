#ifndef TINYNN_NET_H_
#define TINYNN_NET_H_

#include "layer.hpp"
#include <iostream>
#include <unordered_map>

namespace tinynn
{
    class Net
    {
    public:
        Net() : len(0) {}
        ~Net() {}

        Tensor Forward(const Tensor &input)
        {
            Tensor out(input);
            for (int i = 0; i < layers.size(); ++i)
            {
                out = layers[i]->Forward(out);
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

        std::vector<std::pair<Tensor *, Tensor *>> &Params()
        {
            if (len == 0)
            {
                for (auto &layer : layers)
                {
                    auto &param = layer->Params();
                    for (auto name : layer->ParamName())
                    {
                        ++len;
                    }
                }
                params.resize(len);
            }

            int idx = 0;
            for (auto &layer : layers)
            {
                auto &param = layer->Params();
                for (auto name : layer->ParamName())
                {
                    params[idx] = std::make_pair(&param[name].param, &param[name].grad);
                    ++idx;
                }
            }

            return params;
        }

        void Print()
        {
            std::cout << "Net Structure:" << std::endl;
            for (int i = 0; i < layers.size(); ++i)
            {
                std::cout << "layer " << i << ":" << std::endl;
                layers[i]->Print();
            }
        }

        int Save(std::string filename)
        {
            return 0;
        }

        int Load(std::string filename)
        {
            return 0;
        }

        std::vector<std::unique_ptr<Layer>> layers;

    private:
        int len;
        std::vector<std::pair<Tensor *, Tensor *>> params;
    };
}

#endif
