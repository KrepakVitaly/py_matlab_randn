#ifndef VSL_RANDOM_H
#define VSL_RANDOM_H

#include "mkl_vsl.h"
#include "omp.h"

class VSLRandom
{
public:
    VSLRandom();
    ~VSLRandom();
    void rng(int seed);
    void rand(int size, double *arr);
    void randn(int size, double *arr);
private:
    VSLStreamStatePtr *streams;
    int num_procs;
};

#endif // VSL_RANDOM_H
