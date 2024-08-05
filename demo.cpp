#include <stdio.h>
#include <iostream>
#include <immintrin.h>
#include <cstdlib>
#include "voltage.hpp"
#include <math.h>
using namespace std;

int main(){
int size = 640000000;
float* ap = (float*) aligned_alloc(32, size*sizeof(float));
float* bp = (float*) aligned_alloc(32, size*sizeof(float));
if(!ap||!bp){
     cout << "couldnt allocate memory" << endl;
}
float s = 1.5;
 ////vv_dot_product_256(ap,bp,size);
 ////vs_multiply(ap, 2, size);
 ////mv_multiply_256(ap, bp, 4, 3, 4);
 ////mm_multiply_256(ap,bp, 6, 2, 2, 4);
 //for (int i = 0; i < size; i++){
 //        ap[i] = 1;
 //        bp[i] = 1;
 //}
 //cout << vv_dot_avx_cache_optimized(ap,bp,size) << endl;
 //free(ap);
 //free(bp);
 //ap = (float*) aligned_alloc(32, size*sizeof(float));
 //bp = (float*) aligned_alloc(32, size*sizeof(float));
 //if(!ap||!bp){
 //     cout << "couldnt allocate memory" << endl;
 //}
 //for (int i = 0; i < size; i++){
 //        ap[i] = 1;
 //        bp[i] = 1;
 //}
 //cout << dot(ap,bp,size) << endl;
 //free(ap);
 //free(bp);
 //ap = (float*) aligned_alloc(32, size*sizeof(float));
 //bp = (float*) aligned_alloc(32, size*sizeof(float));
 //if(!ap||!bp){
 //     cout << "couldnt allocate memory" << endl;
 //}
 //for (int i = 0; i < size; i++){
 //        ap[i] = 1;
 //        bp[i] = 1;
 //}
 //cout << dot_improved_4(ap,bp,size) << endl;
 //free(ap);
 //free(bp);
 //ap = (float*) aligned_alloc(32, size*sizeof(float));
 //bp = (float*) aligned_alloc(32, size*sizeof(float));
 //if(!ap||!bp){
 //     cout << "couldnt allocate memory" << endl;
 //}
 //for (int i = 0; i < size; i++){
 //        ap[i] = 1;
 //        bp[i] = 1;
 //}
 //cout << dot_improved_2(ap,bp,size) << endl;
 //free(ap);
 //free(bp);
 ap = (float*) aligned_alloc(32, size*sizeof(float));
 float scalar = 1.5;
 if(!ap){
     cout << "couldnt alloc" << endl;
 }
 for (int i = 0; i < size; i++){
     ap[i] = 1;
 }
 vs_multiply(ap, scalar, size);
 vs_multiply_aligned(ap, scalar, size);
 vs_multiply_aligned_2(ap, scalar, size);
 free(ap);
 return 0;
}
