#include <stdio.h>
#include <iostream>
#include <immintrin.h>
#include <cstdlib>
#include "voltage.hpp"
#include <math.h>
using namespace std;

int main(){
int size = 8;
   float* ap = new(nothrow) float[1000000000]; 
   float* bp = new(nothrow) float[1000000000];
   if(!ap||!bp){
        cout << "couldnt allocate memory" << endl;
   } 
   float s = 1.5;
   cout << "populating dummy matrices... " << endl;
    //vv_dot_product_256(ap,bp,size);
    //vs_multiply(ap, 2, size);
    //mv_multiply_256(ap, bp, 4, 3, 4);
    //mm_multiply_256(ap,bp, 6, 2, 2, 4);
    for (int i = 0; i < 1000; i++){
        for (int j = 0; j < 1000; j ++){
            ap[i*1000 + j] = 1;
            bp[i*1000 + j] = 1;
        }
    }
    cout << "done!" << endl;
    mm_multiply_256(ap,bp,1000,1000,1000,1000);
    free(ap); 
    free(bp);
    return 0;
}