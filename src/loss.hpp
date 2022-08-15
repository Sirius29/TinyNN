#ifndef TINYNN_LOSS_H_
#define TINYNN_LOSS_H_

#include "tensor.hpp"

namespace tinynn
{
    template <typename Tp>
    inline Tp Sign(Tp val);

    class LossFunc
    {
    public:
        virtual ~LossFunc() {}
        virtual float ComputeLoss(const Tensor &pred, const Tensor &target) = 0;
        virtual Tensor Grad(const Tensor &pred, const Tensor &target) = 0;
    };

    class MSE : public LossFunc
    {
    public:
        float ComputeLoss(const Tensor &pred, const Tensor &target) override;
        Tensor Grad(const Tensor &pred, const Tensor &target) override;
    };

    class MAE : public LossFunc
    {
    public:
        float ComputeLoss(const Tensor &pred, const Tensor &target) override;
        Tensor Grad(const Tensor &pred, const Tensor &target) override;
    };
}

#endif
