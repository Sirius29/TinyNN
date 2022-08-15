#include "tensor.hpp"
#include <iostream>

namespace tinynn
{
    Tensor Dot(const Tensor &a, const Tensor &b)
    {
        Size a_size = a.GetSize();
        Size b_size = b.GetSize();
        if (a_size.w != b_size.h)
        {
            std::cout << "dimension not match." << std::endl;
            return Tensor();
        }

        Tensor out(Size(a_size.h, b_size.w, 1));
        for (int i = 0; i < a_size.h; ++i)
        {
            float *p_a = a.GetRow<float>(i);
            float *p_out = out.GetRow<float>(i);
            for (int j = 0; j < b_size.w; ++j)
            {
                float sum = 0;
                for (int k = 0; k < a_size.w; ++k)
                {
                    float *p_b = b.GetRow<float>(k);
                    sum += p_a[k] * p_b[j];
                }
                p_out[j] = sum;
            }
        }

        return out;
    }

    Tensor Add(const Tensor &a, const Tensor &b)
    {
        Size a_size = a.GetSize();
        Size b_size = b.GetSize();
        if (a_size != b_size)
        {
            std::cout << "dimension not match." << std::endl;
            return Tensor();
        }

        Tensor out(a_size);
        for (int i = 0; i < a_size.h; ++i)
        {
            float *p_a = a.GetRow<float>(i);
            float *p_out = out.GetRow<float>(i);
            float *p_b = b.GetRow<float>(i);
            for (int j = 0; j < b_size.w; ++j)
            {
                p_out[j] = p_a[j] + p_b[j];
            }
        }

        return out;
    }
}
