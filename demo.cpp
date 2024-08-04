#include <stdio.h>
#include <iostream>
#include <immintrin.h>
#include <cstdlib>
#include "voltage.hpp"
#include <math.h>
using namespace std;

int main(){
int size = 64000;
float* ap = new(nothrow) float[size];
   float* bp = new(nothrow) float[size];
   if(!ap||!bp){
        cout << "couldnt allocate memory" << endl;
   }
   float s = 1.5;
   cout << "populating dummy matrices... " << endl;
    //vv_dot_product_256(ap,bp,size);
    //vs_multiply(ap, 2, size);
    //mv_multiply_256(ap, bp, 4, 3, 4);
    //mm_multiply_256(ap,bp, 6, 2, 2, 4);
    for (int i = 0; i < size; i++){
            ap[i] = 1;
            bp[i] = 1;
    }
    cout << vv_dot_avx_cache_optimized(ap,bp,size) << endl;
    cout << dot(ap,bp,size) << endl;
    free(ap);
    free(bp);
    return 0;
}
