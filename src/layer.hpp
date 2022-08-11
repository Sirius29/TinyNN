#ifndef TINYNN_LAYER_H_
#define TINYNN_LAYER_H_

#include "tensor.hpp"
#include "initializer.hpp"
#include <memory>

namespace tinynn
{
    class Layer
    {
    public:
        virtual ~Layer() {}
        virtual Tensor Forward(const Tensor &_input) = 0;
        virtual Tensor Backward(const Tensor &_grad) = 0;
    };

    class Dense : public Layer
    {
    public:
        Dense(int _out, Initializer *_w_init, Initializer *_b_init) {}
        ~Dense() {}
        Tensor Forward(const Tensor &_input)
        {
            if (!is_init)
            {
                w = w_init->Init(Size(input.GetSize().w, out, 1));
                b = b_init->Init(Size(1, out, 1));
            }
            input = _input;
            return Add<float>(Dot<float>(input, w), b);
        }

        Tensor Backward(const Tensor &_grad)
        {
            w_grad = Dot<float>(input.T<float>(), w);
            b_grad = _grad;
            return Dot<float>(_grad, w.T<float>());
        }

    private:
        int out;
        bool is_init;
        bool is_training;
        std::shared_ptr<Initializer> w_init;
        std::shared_ptr<Initializer> b_init;
        Tensor w;
        Tensor b;
        Tensor input;
        Tensor w_grad;
        Tensor b_grad;
    };
}

#endif
