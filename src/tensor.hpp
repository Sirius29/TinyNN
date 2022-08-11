#ifndef TINYNN_TENSER_H_
#define TINYNN_TENSER_H_

#include "allocator.hpp"
#include <cstdlib>
#include <vector>
#include <string.h>

namespace tinynn
{
    struct Size
    {
        Size() : h(0), w(0), c(0), total(0) {}
        Size(int _h, int _w, int _c) : h(_h), w(_w), c(_c)
        {
            total = h * w * c;
        }

        bool operator==(const Size &size)
        {
            return size.h == h && size.w == w && size.c == c;
        }

        bool operator!=(const Size &size)
        {
            return !(*this == size);
        }

        int h;
        int w;
        int c;
        int total;
    };

    class Tensor
    {
    public:
        Tensor() : size(Size()), element_size(0), allocator(nullptr), data(nullptr) {}
        Tensor(Size _size, size_t _element_size = 4, Allocator *_allocator = nullptr) : size(_size), element_size(_element_size),
                                                                                        allocator(_allocator), data(nullptr)
        {
            if (allocator)
            {
                data = allocator->Malloc(size.total * element_size);
            }
            else
            {
                data = Malloc(size.total * element_size);
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

            size = Size();
        }

        Tensor(const Tensor &t)
        {
            size = t.size;
            element_size = t.element_size;
            allocator = t.allocator;
            memcpy(data, t.data, size.total * element_size);
        }

        Tensor &operator=(const Tensor &t)
        {
            if (this == &t)
            {
                return *this;
            }

            Release();

            size = t.size;
            element_size = t.element_size;
            allocator = t.allocator;
            if (!t.Empty())
            {
                if (allocator)
                {
                    data = allocator->Malloc(size.total * element_size);
                }
                else
                {
                    data = Malloc(size.total * element_size);
                }

                memcpy(data, t.data, size.total * element_size);
            }

            return *this;
        }

        Tensor(Tensor &&t)
        {
            size = t.size;
            element_size = t.element_size;
            data = t.data;
            allocator = t.allocator;

            t.data = nullptr;
            t.size = Size();
        }

        Tensor &operator=(Tensor &&t)
        {
            if (this == &t)
            {
                return *this;
            }

            Release();

            size = t.size;
            element_size = t.element_size;
            data = t.data;
            allocator = t.allocator;

            t.size = Size();
            t.element_size = 0;
            t.data = nullptr;
            t.allocator = nullptr;

            return *this;
        }

        template <typename Tp>
        Tensor T()
        {
            Tensor out(*this);
            for (int i = 0; i < size.w; ++i)
            {
                Tp *p_out = out.GetRow<Tp>(i);
                for (int j = 0; j < size.h; ++j)
                {
                    Tp *p_this = this->GetRow<Tp>(j);
                    for (int k = 0; k < size.c; ++k)
                    {
                        p_out[j + k] = p_this[i + k];
                    }
                }
            }
            return out;
        }

        bool Empty() const
        {
            return data == nullptr;
        }

        int Total() const
        {
            return size.total;
        }

        Size GetSize() const
        {
            return size;
        }

        template <typename Tp>
        Tp *GetData()
        {
            return reinterpret_cast<Tp *>(data);
        }

        template <typename Tp>
        Tp *GetRow(int row)
        {
            int offset = row * size.w * size.c;
            return reinterpret_cast<Tp *>(reinterpret_cast<char *>(data) + offset);
        }

    private:
        Size size;
        size_t element_size;
        void *data;
        Allocator *allocator;
    };

    template <typename Tp>
    Tensor Dot(const Tensor &a, const Tensor &b);

    template <typename Tp>
    Tensor Add(const Tensor &a, const Tensor &b);
}

#endif
