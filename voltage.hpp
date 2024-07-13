// https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#
#ifndef voltage
#define voltage
#include <stdio.h>
#include <iostream>
#include <immintrin.h>
using namespace std;

//AVX dot product of A and B
//ASSUMES 32ALIGNED 
float vv_dot_product_256(const float* A, const float* B, size_t n){
   __m256 sum = _mm256_setzero_ps(); //initializing the sum register
    //looping over sets of 8 floats from A and B for AVX2 multiplication
    float product = 0;
    size_t i;
    cout << "initiating avx2 vv dotproduct. vector size " << n << endl;
        for (i = 0; i + 7 < n; i+=8){
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
        product += _mm_cvtss_f32(result);
    }
        
        /** in case the vector size isnt a multiple of 8, we add the dot and add the remaining scalars manually. 
        i guess for a perfecting this we can made a dummy vector with 0s in the remaining
        i want to wait to see how i implement class-level data handling and 32-alignment, maybe this will be solved auto then
        **/        
       cout << "avx 2 vv dot product: non8multiple size triggered scalar addition of leftovers" << endl;
       for (int i = n - n%8; i < n; i++){
            product += A[i]*B[i];
       }
       cout << "avx2 vv dot product result: " << product << endl;
       return product;
}
#endif