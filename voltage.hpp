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
    //allocating 32bit-aligned memory to vectors
    //adjusting for non 8 multiple n:
    size_t aligned_n = (n%8==0)?n:(n + (8 - n%8)); 
    float* ap = (float*) aligned_alloc(32, sizeof(float)*aligned_n);  
    float* bp = (float*) aligned_alloc(32, sizeof(float)*aligned_n);  
        if(ap == nullptr || bp == nullptr){
            cout << "avx2 vv dotproduct: failed to allocate aligned memory" << endl;
        }
        else{
            for (int i = 0; i <n; i++) {
                ap[i] = A[i];
                bp[i] = B[i];
                cout << i << " added " << ap[i] << " " << bp[i] << endl;
             }
         }
         //since this is a dot, we can simply append 0s from indices n to aligned_n-1]
        cout << "avx2 vv dotproduct: appending 0s to make size 8-multiple" << endl;
        if (n % 8 != 0){
           for (size_t x = n; x < aligned_n; x++){
                ap[x] = 0.0f;
                bp[x] = 0.0f;
           } 
        }
    //calculation
   __m256 sum = _mm256_setzero_ps(); //initializing the sum register
    //looping over sets of 8 floats from A and B for AVX2 multiplication
    float product = 0;
    size_t i;
    cout << "initiating avx2 vv dotproduct. vector size " << n << endl;
        for (i = 0; i < aligned_n; i+=8){
        __m256 sum = _mm256_setzero_ps();
        __m256 va = _mm256_load_ps(&ap[i]);
        __m256 vb = _mm256_load_ps(&bp[i]);
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
    free(ap);
    free(bp);       
    cout << "avx2 vv dot product result: " << product << endl;
    return product;
}

float vs_mult(const float* v, const float* s, size_t size){
    return 0.0f;
}
#endif