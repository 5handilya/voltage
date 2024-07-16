// https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#
#ifndef voltage
#define voltage
#include <stdio.h>
#include <iostream>
#include <immintrin.h>
#include <chrono>
using namespace std;

/** ============================
Voltage, a C++ BLAS by Kunal Shandilya
    Legend for functions etc.:
        v = vector
        s = scalar
        m = matrix    

 ============================ */
//AVX dot product of A and B
//also handles 32bit alignment
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

float* vs_multiply(const float* v, const float s, size_t size){
    auto fullstart = chrono::system_clock().now();
    cout << "avx2 vs mult: multiplying vector of size " << size << " with scalar " << s << endl; 
    size_t aligned_size = (size%8==0)?size:(size + (8 - size%8));
    //allocating 32-bit aligned memory
    float* v32 = (float*) aligned_alloc(32, sizeof(float)*aligned_size);
    __m256 s32reg = _mm256_set1_ps(s);
    size_t i;
    for (i = 0; i < size; i++){
        v32[i] = v[i];
    }
    //adjustment for non 8 multiple sized
    if (size != aligned_size){
        cout << "avx2 vs mult: appending 0s to increase vector size to 8-multiple" << endl;
        //append 0s to vector, trim to size before returning
        size_t x;
        for (x = size; x < aligned_size; x++){
            v32[x] = 0.0f;
        }
    }
    float* result = (float*) aligned_alloc(32, sizeof(float)*aligned_size);
    //multiplication
    size_t c;
    auto multstart = chrono::system_clock().now();
    for (c = 0; c < aligned_size; c+=8){
        __m256 v32reg = _mm256_load_ps(&v32[c]);
        __m256 product = _mm256_mul_ps(v32reg, s32reg);
        _mm256_store_ps(&result[c], product);
    }
    auto multstop = chrono::system_clock().now();
    float* trimmed_result = result;
    if (size != aligned_size){
        size_t j;
        for (j = 0; j < size; j++){
            trimmed_result[j] = result[j];
        }
    }
    free(v32);
    cout << "debug vs mult result:"<< endl;
    cout << " { ";
    for ( int i = 0; i < size;  i++){
        cout << result[i] << " ";
    }
    cout << "}" << endl;
    auto fullstop = chrono::system_clock().now();
    cout << "avx vs mult: result calculated" << endl;
    cout << "debug stats: " << endl << "full time (microseconds): " << chrono::duration_cast<chrono::microseconds>(fullstop-fullstart).count() << endl << " mult time (microseconds): " << chrono::duration_cast<chrono::microseconds>(multstop-multstart).count() << endl;  
    return trimmed_result;
}

/**TODO
1. Transpose vectorized
2. Matrix add, port existing code here
3. Matrix class, port existing code here  
*/
#endif