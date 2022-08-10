#ifndef TINYNN_TENSER_H_
#define TINYNN_TENSER_H_

#include "allocator.hpp"
#include <vector>

class Tensor
{
public:
    Tensor() {}
    Tensor(int _h, int _w, int _c, size_t _element_size, Allocator *_allocator = nullptr) : h(_h), w(_w), c(_c), element_size(_element_size)
    {
        allocator = _allocator;
        total_size = h * w * c;
        if (allocator)
        {
            data = allocator->Malloc(total_size * element_size);
        }
        else
        {
            data = Malloc(total_size * element_size);
        }
    }

    ~Tensor()
    {
        Release();
    }

    void Release()
    {
        if (data != nullptr)
        {
            if (allocator)
            {
                allocator->Free(data);
            }
            else
            {
                Free(data);
            }
            data = nullptr;
        }

        h = 0;
        w = 0;
        c = 0;
        total_size = 0;
    }

    Tensor(Tensor &&t)
    {
        w = t.w;
        h = t.h;
        c = t.c;
        element_size = t.element_size;
        allocator = t.allocator;
        total_size = t.total_size;
        data = t.data;

        t.data = nullptr;
    }

    Tensor &operator=(Tensor &&t)
    {
        w = t.w;
        h = t.h;
        c = t.c;
        element_size = t.element_size;
        allocator = t.allocator;
        total_size = t.total_size;
        data = t.data;
        
        t.data = nullptr;

        return *this;
    }

    template <typename Tp>
    Tp *GetData()
    {
        return reinterpret_cast<Tp *>(data);
    }

private:
    int h;
    int w;
    int c;
    size_t element_size;
    Allocator *allocator;
    int total_size;
    void *data;
};

#endif
