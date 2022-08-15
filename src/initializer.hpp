#ifndef TINYNN_INITIALIZER_H_
#define TINYNN_INITIALIZER_H_

#include "tensor.hpp"
#include <random>

namespace tinynn
{
    class Initializer
    {
    public:
        virtual ~Initializer() {}
        virtual Tensor Init(Size size) = 0;
    };

    class Normal : public Initializer
    {
    public:
        Normal(float _mean = 0.f, float _std = 1.f) : distribution(_mean, _std) {}
        Tensor Init(Size size) override
        {
            Tensor t(size);
            float *p_data = t.GetData<float>();
            for (int i = 0; i < size.total; ++i)
            {
                p_data[i] = distribution(gen);
            }
            return t;
        }

    private:
        std::normal_distribution<float> distribution;
        std::mt19937 gen;
    };

    class TruncatedNormal : public Initializer
    {
    public:
        TruncatedNormal(float _low, float _high, float _mean = 0.f, float _std = 1.f) : low(_low), high(_high), distribution(_mean, _std) {}
        Tensor Init(Size size) override
        {
            Tensor t(size);
            float *p_data = t.GetData<float>();
            for (int i = 0; i < size.total; ++i)
            {
                float val = distribution(gen);
                while (true)
                {
                    if (val > low && val < high)
                    {
                        break;
                    }
                    val = distribution(gen);
                }
                p_data[i] = val;
            }
            return t;
        }

    private:
        float low;
        float high;
        std::normal_distribution<float> distribution;
        std::mt19937 gen;
    };

    class Uniform : public Initializer
    {
    public:
        Uniform(float a = 0.f, float b = 1.f) : distribution(a, b) {}
        Tensor Init(Size size) override
        {
            Tensor t(size);
            float *p_data = t.GetData<float>();
            for (int i = 0; i < size.total; ++i)
            {
                p_data[i] = distribution(gen);
            }
            return t;
        }

    private:
        std::uniform_real_distribution<float> distribution;
        std::mt19937 gen;
    };

    class Constant : public Initializer
    {
    public:
        Constant(float _val) : val(_val) {}
        Tensor Init(Size size) override
        {
            Tensor t(size);
            float *p_data = t.GetData<float>();
            for (int i = 0; i < size.total; ++i)
            {
                p_data[i] = val;
            }
            return t;
        }

    private:
        float val;
    };

    class Zeros : public Constant
    {
    public:
        Zeros() : Constant(0.f) {}
    };

    class Ones : public Constant
    {
    public:
        Ones() : Constant(1.f) {}
    };

}

#endif
