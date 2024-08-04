// https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#
/**
░▒▓█▓▒░░▒▓█▓▒░░▒▓██████▓▒░░▒▓█▓▒░   ░▒▓████████▓▒░▒▓██████▓▒░ ░▒▓██████▓▒░░▒▓████████▓▒░
░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░  ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░
 ░▒▓█▓▒▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░  ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░
 ░▒▓█▓▒▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░  ░▒▓████████▓▒░▒▓█▓▒▒▓███▓▒░▒▓██████▓▒░
  ░▒▓█▓▓█▓▒░ ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░  ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░
  ░▒▓█▓▓█▓▒░ ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░  ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░
   ░▒▓██▓▒░   ░▒▓██████▓▒░░▒▓████████▓▒░▒▓█▓▒░  ░▒▓█▓▒░░▒▓█▓▒░░▒▓██████▓▒░░▒▓████████▓▒░

VOLTAGE, a C++ BLAS by Kunal Shandilya
Legend for functions etc.:
    v = vector
    s = scalar
    m = matrix

============================ */
#ifndef voltage
#define voltage
#include <stdio.h>
#include <iostream>
#include <immintrin.h>
#include <chrono>
using namespace std;
using namespace std::chrono;

class matrix{
    public:
    private:

};
//AVX dot product of A and B
//also handles 32bit alignment
float vv_dot_avx_cache_optimized(const float* a, const float* b, size_t n) {
    constexpr size_t cache_line = 64; //cache line size in bytes
    constexpr size_t floats_per_cache_line = cache_line / sizeof(float);
    auto start = system_clock().now();
    __m256 sum = _mm256_setzero_ps();
    for (size_t i = 0; i < n; i += floats_per_cache_line) {
        for (size_t j = i; j < i + floats_per_cache_line; j++) {
            __m256 va = _mm256_loadu_ps(&a[j]);
            __m256 vb = _mm256_loadu_ps(&b[j]);
            sum = _mm256_add_ps(sum, _mm256_mul_ps(va, vb));
        }
    }
    __m256 hsum = _mm256_hadd_ps(sum, sum);
    hsum = _mm256_hadd_ps(hsum, hsum);
    auto stop = system_clock().now();
    cout << "vv dot avx cache optim. duration (us): " << duration_cast<microseconds>(stop - start).count() << endl;
    cout << "vv dot. cache optim flops: " << duration_cast<microseconds>(stop - start).count()*1000000/(2000) << endl;
    return _mm_cvtss_f32(_mm256_extractf128_ps(hsum, 0)) + _mm_cvtss_f32(_mm256_extractf128_ps(hsum, 1));
}
float dot(const float* a, const float* b, size_t n) {
    auto start = system_clock().now();
    __m256 sum = _mm256_setzero_ps();
    for (size_t i = 0; i < n; i += 8) {
        for (size_t j = i; j < i+8; j++) {
            __m256 va = _mm256_loadu_ps(&a[j]);
            __m256 vb = _mm256_loadu_ps(&b[j]);
            sum = _mm256_fmadd_ps(va, vb, sum);
        }
    }
    __m256 hsum = _mm256_hadd_ps(sum, sum);
    hsum = _mm256_hadd_ps(hsum, hsum);
    auto stop = system_clock().now();
    cout << "vv dot. duration (us): " << duration_cast<microseconds>(stop - start).count() << endl;
    cout << "vv dot. flops: " << duration_cast<microseconds>(stop - start).count()*1000000/(2000) << endl;
    return _mm_cvtss_f32(_mm256_extractf128_ps(hsum, 0)) + _mm_cvtss_f32(_mm256_extractf128_ps(hsum, 1));
}
float vv_dot_product_256(float* A, float* B, size_t n){
    //allocating 32bit-aligned memory to vectors
    //adjusting for non 8 multiple n:
    size_t aligned_n = (n%8==0)?n:(n + (8 - n%8));
    //float* ap = (float*) aligned_alloc(32, sizeof(float)*aligned_n);
    //float* bp = (float*) aligned_alloc(32, sizeof(float)*aligned_n);
    //    if(ap == nullptr || bp == nullptr){
    //        cout << "avx2 vv dotproduct: failed to allocate aligned memory" << endl;
    //    }
    //    else{
    //        for (int i = 0; i <n; i++) {
    //            ap[i] = A[i];
    //            bp[i] = B[i];
    //            cout << i << " added " << ap[i] << " " << bp[i] << endl;
    //         }
    //     }
         //since this is a dot, we can simply append 0s from indices n to aligned_n-1]
        cout << "avx2 vv dotproduct: appending 0s to make size 8-multiple" << endl;
        if (n % 8 != 0){
           for (size_t x = n; x < aligned_n; x++){
                A[x] = 0.0f;
                B[x] = 0.0f;
           }
        }
    //calculation
   __m256 sum = _mm256_setzero_ps(); //initializing the sum register
    //looping over sets of 8 floats from A and B for AVX2 multiplication
    float product = 0;
    size_t i;
    cout << "initiating avx2 vv dotproduct. vector size " << n << endl;
        for (i = 0; i < aligned_n; i+=8){
       // __m256 sum = _mm256_setzero_ps();
       // __m256 va = _mm256_load_ps(&A[i]);
       // __m256 vb = _mm256_load_ps(&B[i]);
        //__m256 mul = _mm256_mul_ps(va, vb);
        __m256 sum = _mm256_mul_ps(_mm256_load_ps(&A[i]), _mm256_load_ps(&B[i]));
        //sum = _mm256_add_ps(sum, mul);

        //next, we need to add the partial sums inside the 'sum' register by consecutive horizontal adds
        __m256 hsum = _mm256_hadd_ps(sum, sum);
        hsum = _mm256_hadd_ps(hsum, hsum);
        __m128 bottomhalf = _mm256_castps256_ps128(hsum);
        __m128 tophalf = _mm256_extractf128_ps(hsum, 1); //interestingly, extractf128 position starts from 1, not 0
        __m128 result = _mm_add_ps(bottomhalf, tophalf);
        product += _mm_cvtss_f32(result);
    }
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
