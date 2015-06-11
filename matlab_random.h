#include <cmath>
#include <cstring>

class MatlabRandn
{
public:
    MatlabRandn();
    void rng(long long seed);
    void randn(long long size, double *arr);
private:
    void genrand_int_vector(unsigned int u[2]);
    double genrandu();
    unsigned int mt[625];
};
