#ifndef TINYNN_ALLOCATOR_H_
#define TINYNN_ALLOCATOR_H_

#include <memory>

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
        void *Malloc(size_t size) override
        {
            void *p = malloc(size);
            return p;
        }

        void Free(void *p) override
        {
            free(p);
        }
    };
}

#endif
