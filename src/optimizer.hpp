#ifndef TINYNN_OPTIMIZER_H_
#define TINYNN_OPTIMIZER_H_

#include "tensor.hpp"

namespace tinynn
{
    class Optimizer
    {
    public:
        virtual ~Optimizer(){};
        virtual int Step(const std::vector<std::pair<Tensor *, Tensor *>> &params) = 0;
    };

    class SGD : public Optimizer
    {
    public:
        SGD(float _lr = 0.01f, float _weight_decay = 0.f) : lr(_lr), weight_decay(_weight_decay) {}

        int Step(const std::vector<std::pair<Tensor *, Tensor *>> &params) override
        {
            for (std::pair<Tensor *, Tensor *> param : params)
            {
                Size size = param.first->GetSize();
                for (int i = 0; i < size.h; ++i)
                {
                    float *param_row = param.first->GetRow<float>(i);
                    float *grad_row = param.second->GetRow<float>(i);
                    for (int j = 0; j < size.w; ++j)
                    {
                        param_row[j] += -lr * grad_row[j];
                    }
                }
            }

            return 0;
        }

    private:
        float lr;
        float weight_decay;
    };

    class Adam : public Optimizer
    {
    public:
        Adam(float _lr = 0.001f, float _beta1 = 0.9f, float _beta2 = 0.999f, float _epsilon = 1e-8, float _weight_decay = 0.f) : lr(_lr), b1(_beta1), b2(_beta2),
                                                                                                                                 epsilon(_epsilon), weight_decay(_weight_decay)
        {
            t = 0.f;
            m = 0.f;
            v = 0.f;
        }

        int Step(const std::vector<std::pair<Tensor *, Tensor *>> &params) override
        {
            t += 1;
            // m += (1.0 - b1) * (grads - m);
            // v += (1.0 - b2) * (grads ** 2 - v);
            for (std::pair<Tensor *, Tensor *> param : params)
            {
                Size size = param.first->GetSize();
                for (int i = 0; i < size.h; ++i)
                {
                    float *param_row = param.first->GetRow<float>(i);
                    float *grad_row = param.second->GetRow<float>(i);
                    for (int j = 0; j < size.w; ++j)
                    {
                        param_row[j] += -lr * grad_row[j];
                    }
                }
            }
            return 0;
        }

    private:
        float lr;
        float b1;
        float b2;
        float epsilon;
        float weight_decay;
        int t;
        float m;
        float v;
    };

    class Momentum : public Optimizer
    {
    public:
        Momentum(float _lr, float _momentum, float _weight_decay = 0.f) : lr(_lr), momentum(_momentum), weight_decay(_weight_decay)
        {
            acc = 0;
        }

        int Step(const std::vector<std::pair<Tensor *, Tensor *>> &params) override
        {
            for (std::pair<Tensor *, Tensor *> param : params)
            {
                Size size = param.first->GetSize();
                for (int i = 0; i < size.h; ++i)
                {
                    float *param_row = param.first->GetRow<float>(i);
                    float *grad_row = param.second->GetRow<float>(i);
                    for (int j = 0; j < size.w; ++j)
                    {
                        param_row[j] += -lr * grad_row[j];
                    }
                }
            }
            return 0;
        }

    private:
        float lr;
        float momentum;
        float weight_decay;
        float acc;
    };
}

#endif