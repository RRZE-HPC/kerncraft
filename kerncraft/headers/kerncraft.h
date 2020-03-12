#ifndef _KERNCRAFT_H
#define _KERNCRAFT_H
#include <stddef.h>
#include <stdlib.h>

#define swap(a, b) {void* t = a; a = b; b = t;}


inline void* aligned_malloc(size_t size, size_t align) {
    // Based on http://stackoverflow.com/q/16376942
    void *result;
    #if defined(_MSC_VER)
        result = _aligned_malloc(size, align);
    #elif defined(__INTEL_COMPILER)
         result = _mm_malloc(size, align);
    #else
         if(posix_memalign(&result, align, size)) result = 0;
    #endif
    return result;
};

#endif
