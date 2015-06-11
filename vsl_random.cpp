#include "vsl_random.h"

VSLRandom::VSLRandom()
{
    num_procs = omp_get_num_procs();
    streams = new VSLStreamStatePtr[num_procs];
    for (int i = 0; i < num_procs; ++i) {
        vslNewStream(&streams[i], VSL_BRNG_MT2203+i, 0);
    }
}

VSLRandom::~VSLRandom()
{
    for (int i = 0; i < num_procs; ++i) {
        vslDeleteStream(&streams[i]);
    }
    delete[] streams;
}

void VSLRandom::rng(int seed)
{
    for (int i = 0; i < num_procs; ++i) {
        vslDeleteStream(&streams[i]);
        vslNewStream(&streams[i], VSL_BRNG_MT2203+i, seed);
    }
    return;   
}

void VSLRandom::rand(int size, double *arr)
{
    #pragma omp parallel
    {
        int block_size;
        int thread_id = omp_get_thread_num();
        int start_idx = thread_id * (size / num_procs);
        if (num_procs - 1 == thread_id) {
            block_size = size - start_idx;
        } else {
            block_size = size / num_procs;
        }
        vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, streams[thread_id], block_size, arr+start_idx, 0, 1);
    }
    return;
}

void VSLRandom::randn(int size, double *arr)
{
    #pragma omp parallel
    {
        int block_size;
        int thread_id = omp_get_thread_num();
        int start_idx = thread_id * (size / num_procs);
        if (num_procs - 1 == thread_id) {
            block_size = size - start_idx;
        } else {
            block_size = size / num_procs;
        }
        vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, streams[thread_id], block_size, arr+start_idx, 0.0, 1.0);
    }
    return;
}
