#ifndef TINYNN_LOSS_H_
#define TINYNN_LOSS_H_

#include "tensor.hpp"

namespace tinynn
{
    class Loss
    {
    public:
        virtual ~Loss() {}
        virtual Tensor Forward(const Tensor &pred, const Tensor &labal) = 0;
        virtual Tensor Backward(const Tensor &pred, const Tensor &labal) = 0;
    };

    class MSE : public Loss
    {
    public:
    };
}

#endif
