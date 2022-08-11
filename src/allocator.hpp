#ifndef TINYNN_ALLOCATOR_H_
#define TINYNN_ALLOCATOR_H_

#include <memory>
#include <stddef.h>

namespace tinynn
{
    void *Malloc(size_t size);

    void Free(void *p);

    class Allocator
    {
    public:
        virtual ~Allocator();
        virtual void *Malloc(size_t size) = 0;
        virtual void Free(void *ptr) = 0;
    };

    class GeneralAlloc : public Allocator
    {
    public:
        GeneralAlloc() {}
        ~GeneralAlloc() {}
        void *Malloc(size_t size)
        {
            void *p = malloc(size);
            return p;
        }

        void Free(void *p)
        {
            free(p);
        }
    };
}

#endif
