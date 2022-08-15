#include "loss.hpp"
#include <iostream>

namespace tinynn
{
    template <typename Tp>
    inline Tp Sign(Tp val)
    {
        if (val < 0)
        {
            return -1;
        }
        else if (val == 0)
        {
            return 0;
        }
        else
        {
            return 1;
        }
    }

    float MSE::ComputeLoss(const Tensor &pred, const Tensor &target)
    {
        Size pred_size = pred.GetSize();
        Size target_size = target.GetSize();
        if (pred_size != target_size)
        {
            std::cout << "dimension is not match." << std::endl;
        }

        float ComputeLoss = 0.f;
        for (int i = 0; i < pred_size.h; ++i)
        {
            float *pred_data = pred.GetRow<float>(i);
            float *target_data = target.GetRow<float>(i);
            for (int j = 0; j < pred_size.w; ++j)
            {
                float diff = pred_data[j] - target_data[j];
                ComputeLoss += diff * diff;
                // for(int k = 0; k < pred_size.c; ++k) // channel todo
                // {
                //     ComputeLoss += pred_data[j];
                // }
            }
        }
        return ComputeLoss * 0.5 / pred_size.w;
    }

    Tensor MSE::Grad(const Tensor &pred, const Tensor &target)
    {
        Size pred_size = pred.GetSize();
        Size target_size = target.GetSize();
        if (pred_size != target_size)
        {
            std::cout << "dimension is not match." << std::endl;
        }

        Tensor gred(pred_size);
        for (int i = 0; i < pred_size.h; ++i)
        {
            float *pred_data = pred.GetRow<float>(i);
            float *target_data = target.GetRow<float>(i);
            float *gred_data = gred.GetRow<float>(i);
            for (int j = 0; j < pred_size.w; ++j)
            {
                gred_data[j] = (pred_data[j] - target_data[j]) / pred_size.w;
            }
        }

        return gred;
    }

    float MAE::ComputeLoss(const Tensor &pred, const Tensor &target)
    {
        Size pred_size = pred.GetSize();
        Size target_size = target.GetSize();
        if (pred_size != target_size)
        {
            std::cout << "dimension is not match." << std::endl;
        }

        float ComputeLoss = 0.f;
        for (int i = 0; i < pred_size.h; ++i)
        {
            float *pred_data = pred.GetRow<float>(i);
            float *target_data = target.GetRow<float>(i);
            for (int j = 0; j < pred_size.w; ++j)
            {
                ComputeLoss += abs(pred_data[j] - target_data[j]);
                // for(int k = 0; k < pred_size.c; ++k) // channel todo
                // {
                //     ComputeLoss += pred_data[j];
                // }
            }
        }
        return ComputeLoss / pred_size.w;
    }

    Tensor MAE::Grad(const Tensor &pred, const Tensor &target)
    {
        Size pred_size = pred.GetSize();
        Size target_size = target.GetSize();
        if (pred_size != target_size)
        {
            std::cout << "dimension is not match." << std::endl;
        }

        Tensor gred(pred_size);
        for (int i = 0; i < pred_size.h; ++i)
        {
            float *pred_data = pred.GetRow<float>(i);
            float *target_data = target.GetRow<float>(i);
            float *gred_data = gred.GetRow<float>(i);
            for (int j = 0; j < pred_size.w; ++j)
            {
                gred_data[j] = Sign(pred_data[j] - target_data[j]) / pred_size.w;
            }
        }

        return gred;
    }
}
