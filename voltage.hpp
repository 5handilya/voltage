// https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#

#include <stdio.h>
#include <iostream>
#include <immintrin.h>
using namespace std;

//AVX dot product of A and B
float vv_dot_product_256(const float* A, const float* B, size_t n){
   __m256 sum = _mm256_setzero_ps(); //initializing the sum register
    //looping over sets of 8 floats from A and B for AVX2 multiplication
    //TODO: function for adjusting vector to be 1. aligned memory 2. stored in 8-multiple sizes - maybe handle these through matrix class
    size_t i;
    for (i = 0; i < n; i+=8){
        __m256 sum = _mm256_setzero_ps();
        __m256 va = _mm256_load_ps(&A[i]);
        __m256 vb = _mm256_load_ps(&B[i]);
        __m256 mul = _mm256_mul_ps(va, vb);
        sum = _mm256_add_ps(sum, mul);

        //next, we need to add the partial sums inside the 'sum' register by consecutive horizontal adds
        __m256 hsum = _mm256_hadd_ps(sum, sum);
        hsum = _mm256_hadd_ps(hsum, hsum);
        __m128 bottomhalf = _mm256_castps256_ps128(hsum);
        __m128 tophalf = _mm256_extractf128_ps(hsum, 1); //interestingly, extractf128 position starts from 1, not 0
        __m128 result = _mm_add_ps(bottomhalf, tophalf);
        
        return _mm_cvtss_f32(result); 
    }
}