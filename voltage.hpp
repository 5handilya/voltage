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
//AVX2 functions
//32bit alignment handling inbuilt
float vv_dot_product_256(const float* A, const float* B, size_t n){
    //allocating 32bit-aligned memory to vectors
    //adjusting for non 8 multiple n:
    size_t aligned_n = (n%8==0)?n:(n + (8 - n%8)); 
    if (n < 8){
        aligned_n = 8;
    }
    float* ap = (float*) aligned_alloc(32, sizeof(float)*aligned_n);  
    float* bp = (float*) aligned_alloc(32, sizeof(float)*aligned_n);  
        if(ap == nullptr || bp == nullptr){
            cout << "avx2 vv dotproduct: failed to allocate aligned memory" << endl;
        }
        else{
            for (int i = 0; i <n; i++) {
                ap[i] = A[i];
                bp[i] = B[i];
             }
         }
         //since this is a dot, we can simply append 0s from indices n to aligned_n-1]
        cout << "avx2 vv dotproduct: appending 0s to make size 8-multiple" << endl;
        if (n != aligned_n){
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
float* vs_multiply_256(const float* v, const float s, size_t size){
    cout << "avx2 vs mult: multiplying vector of size " << size << " with scalar " << s << endl; 
    auto fullstart = chrono::system_clock().now();
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
        //append 0s to vector, trim to size before returning
        size_t x;
        for (x = size; x < aligned_size; x++){
            v32[x] = 0.0f;
        }
    }
    float* result = (float*) aligned_alloc(32, sizeof(float)*aligned_size);
    //multiplication
    auto multstart = chrono::system_clock().now();
    size_t c;
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
    auto fullstop = chrono::system_clock().now();
    cout << "debug vs mult result:"<< endl;
    cout << " { ";
    for ( int i = 0; i < size;  i++){
        cout << result[i] << " ";
    }
    cout << "}" << endl;
    cout << "avx vs mult: result calculated" << endl;
    cout << "debug stats: " << endl;
    cout << "full time (us): " << chrono::duration_cast<chrono::microseconds>(fullstop-fullstart).count() << endl;
    cout << "mult time (us): " << chrono::duration_cast<chrono::microseconds>(multstop-multstart).count() << endl;
    cout << "pre-mult time (us): " << chrono::duration_cast<chrono::microseconds>(multstart-fullstart).count() << endl;
    cout << "post-mult time (us): " << chrono::duration_cast<chrono::microseconds>(fullstop-multstop).count() << endl;
    return trimmed_result;
}
float * mv_multiply_256(const float* matrix, const float* vect, size_t size_mr, size_t size_mc, size_t size_v){
    //create result vector of size rows(m) x 1 with aligned alloc 
    //load vector in register
    //consec load matrix rows
    //m256_mul, store in
    cout << "avx2 mv_multiply: matrix size " << size_mr <<  "  x  " <<  size_mc<< " vector size " << size_v << endl;
    //basically a series of vv_mults
    size_t r,c;
    float* result = (float*) aligned_alloc(32, sizeof(float)*size_mr);
    float* submatrix = (float*) aligned_alloc(32, sizeof(float)*size_mc);
    float subresult;
    for (r = 0; r < size_mr; r ++){
        for (c = 0; c < size_mc; c++){
           submatrix[c] = matrix[r*size_mc + c]; 
        }
        subresult = vv_dot_product_256(submatrix, vect, size_mc);
        result[r] = subresult;
    }
    size_t x;
    cout << "avx2 vv_mult result: " << endl;
    for (x = 0; x < size_mr; x++){
        cout << result[x] << endl;
    }
    return result;
}
float* mm_multiply_256(const float* matrix1, const float* matrix2, size_t size_m, size_t size_n, size_t size_o, size_t size_p){
    //result matrix allocated m * p * size(float)
    //check columns equal
    float* result = (float*) aligned_alloc(32, sizeof(float)*size_m*size_p);
    if(size_n != size_o){
        cout << "check your input you flaky bitch take this nullptr fuck you" << endl;
        return nullptr;
    }
    size_t c2i, r2i;
    float* m2v = (float*)aligned_alloc(32, sizeof(float)*size_o); 
    float* subresult = (float*)aligned_alloc(32, sizeof(float*)*size_m);
    for (c2i = 0; c2i < size_p; c2i ++ ){
        cout << "M2V" << endl;
        for (r2i = 0; r2i < size_o; r2i ++){
            m2v[r2i]  = matrix2[c2i + r2i* size_p]; 
            cout << m2v[r2i] << endl;
        }
        cout << endl;
        subresult = mv_multiply_256(matrix1, m2v, size_m, size_n, size_m);
        for (r2i = 0; r2i < size_m; r2i ++){
            cout <<"subresult " << subresult[r2i] <<  "FOR " << r2i << " x " << c2i <<  " PUT AT " << r2i*size_p + c2i << endl;
            result[r2i*size_p + c2i] = subresult[r2i];
        }
    }
    cout << "AVX2 MM_MULT RESULT: " << endl;
    for (int x = 0; x < size_m; x++){
        for (int y = 0; y < size_p; y++){
            cout << " " << result[x*size_p + y] << " ";
        }
        cout << endl;
    }
    return result;
}

#endif