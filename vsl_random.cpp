#include "vsl_random.h"
#include <iostream>

VSLRandom::VSLRandom(): num_procs(0)
{
    num_procs = omp_get_num_procs();
    streams.resize(num_procs);
    for (int i = 0; i < num_procs; ++i) {
        vslNewStream(&streams[i], VSL_BRNG_MT2203+i, 0);
    }
}

VSLRandom::~VSLRandom()
{
    for (int i = 0; i < num_procs; ++i) {
        vslDeleteStream(&streams[i]);
    }
}

VSLRandom::VSLRandom(const VSLRandom &obj)
{
    for (int i = 0; i < num_procs; ++i) {
        vslDeleteStream(&streams[i]);
    }
    num_procs = obj.num_procs;
    streams.resize(num_procs);
    for (int i = 0; i < num_procs; ++i) {
        vslCopyStream(&streams[i], obj.streams[i]);
    }
}

VSLRandom& VSLRandom::operator=(const VSLRandom &rhs)
{
    for (int i = 0; i < num_procs; ++i) {
        vslDeleteStream(&streams[i]);
    }
    num_procs = rhs.num_procs;
    streams.resize(num_procs);
    for (int i = 0; i < num_procs; ++i) {
        vslCopyStream(&streams[i], rhs.streams[i]);
    }
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
