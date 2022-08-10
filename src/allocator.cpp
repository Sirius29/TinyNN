#include "allocator.hpp"

void *Malloc(size_t size)
{
    void *p = malloc(size);
    return p;
}

void Free(void *p)
{
    free(p);
}
