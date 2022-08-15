#ifndef TINYNN_MODEL_H_
#define TINYNN_MODEL_H_

#include "net.hpp"
#include "loss.hpp"
#include "optimizer.hpp"
#include <assert.h>

namespace tinynn
{
    class Model
    {
    public:
        Model() {}
        ~Model() {}

        Tensor Forward(const Tensor &input)
        {
            return net.Forward(input);
        }

        Tensor Backward(const Tensor &pred, const Tensor &target)
        {
            float loss = loss_func->ComputeLoss(pred, target);
            Tensor grad = loss_func->Grad(pred, target);
            return net.Backward(grad);
        }

        int Update()
        {
            std::vector<std::pair<Tensor *, Tensor *>> &params = net.Params();
            optim->Step(params);
            return 0;
        }

    // private:
        Net net;
        std::unique_ptr<LossFunc> loss_func;
        std::unique_ptr<Optimizer> optim;
    };
}

#endif
