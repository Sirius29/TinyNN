#ifndef TINYNN_LAYER_H_
#define TINYNN_LAYER_H_

#include "tensor.hpp"
#include "initializer.hpp"

#include <iostream>
#include <unordered_map>
#include <algorithm>
#include <memory>

namespace tinynn
{
    struct LayerParam
    {
        Size size;
        Tensor param;
        Tensor grad;
        Initializer *init;
    };

    class Layer
    {
    public:
        virtual ~Layer() {}
        virtual Tensor Forward(const Tensor &_input) = 0;
        virtual Tensor Backward(const Tensor &_grad) = 0;
        virtual void Print() = 0;
        virtual std::unordered_map<std::string, LayerParam> &Params()
        {
            return params;
        }

        virtual std::vector<std::string> ParamName()
        {
            return {};
        }

    protected:
        std::unordered_map<std::string, LayerParam> params;
    };

    class Dense : public Layer
    {
    public:
        Dense(int _out, Initializer *_w_init, Initializer *_b_init) : out(_out), is_init(false), is_training(false)
        {
            params["w"].init = _w_init;
            params["b"].init = _b_init;
        }

        ~Dense() {}

        Tensor Forward(const Tensor &_input) override
        {
            if (!is_init)
            {
                params["w"].size = Size(_input.GetSize().w, out, 1);
                params["b"].size = Size(1, out, 1);
                for (auto p : ParamName())
                {
                    params[p].param = params[p].init->Init(params[p].size);
                }
            }
            input = _input;
            return Add(Dot(input, params["w"].param), params["b"].param);
        }

        Tensor Backward(const Tensor &_grad) override
        {
            params["w"].grad = Dot(input.T<float>(), params["w"].param);
            params["b"].grad = _grad;
            return Dot(_grad, params["w"].param.T<float>());
        }

        void Print() override
        {
        }

        std::vector<std::string> ParamName() override
        {
            return {"w", "b"};
        }

    private:
        int out;
        bool is_init;
        bool is_training;
        Tensor input;
    };

    class ReShape : public Layer
    {
    public:
        ReShape(Size _output) : output(_output) {}
        Tensor Forward(const Tensor &_input)
        {
            input = _input.GetSize();
            return _input;
        }
        Tensor Backward(const Tensor &_grad) override
        {
            return _grad;
        }

    private:
        Size input;
        Size output;
    };

    class Activation : public Layer
    {
    public:
        Activation() {}
        virtual ~Activation() {}
        virtual Tensor Func(const Tensor &_input) = 0;
        virtual Tensor Derivative(const Tensor &_grad) = 0;
        Tensor Forward(const Tensor &_input) override
        {
            input = _input;
            return Func(_input);
        }
        Tensor Backward(const Tensor &_grad) override
        {
            return Dot(Derivative(input), _grad);
        }

    private:
        Tensor input;
    };

    class Sigmoid : public Activation
    {
    public:
        Sigmoid() {}
        Tensor Func(const Tensor &_input) override
        {
            Size size = _input.GetSize();
            Tensor out(size);
            for (int i = 0; i < size.h; ++i)
            {
                float *p_in = _input.GetRow<float>(i);
                float *p_out = out.GetRow<float>(i);
                for (int j = 0; j < size.w; ++j)
                {
                    p_out[j] = 1.f / (1.f + expf(p_in[j]));
                }
            }
            return out;
        }

        Tensor Derivative(const Tensor &_grad) override
        {
            Size size = _grad.GetSize();
            Tensor out(size);
            for (int i = 0; i < size.h; ++i)
            {
                float *p_in = _grad.GetRow<float>(i);
                float *p_out = out.GetRow<float>(i);
                for (int j = 0; j < size.w; ++j)
                {
                    float val = 1.f / (1.f + expf(p_in[j]));
                    p_out[j] = val * (1.f - val);
                }
            }
            return out;
        }

        void Print() override
        {
        }
    };

    class ReLU : public Activation
    {
    public:
        ReLU() {}
        Tensor Func(const Tensor &_input) override
        {
            Size size = _input.GetSize();
            Tensor out(size);
            for (int i = 0; i < size.h; ++i)
            {
                float *p_in = _input.GetRow<float>(i);
                float *p_out = out.GetRow<float>(i);
                for (int j = 0; j < size.w; ++j)
                {
                    p_out[j] = std::max(p_in[j], 0.f);
                }
            }
            return out;
        }

        Tensor Derivative(const Tensor &_grad) override
        {
            Size size = _grad.GetSize();
            Tensor out(size);
            for (int i = 0; i < size.h; ++i)
            {
                float *p_in = _grad.GetRow<float>(i);
                float *p_out = out.GetRow<float>(i);
                for (int j = 0; j < size.w; ++j)
                {
                    p_out[j] = p_in[j] > 0.f;
                }
            }
            return out;
        }

        void Print() override
        {
        }
    };
}

#endif
