#include <stdio.h>
#include <iostream>
#include <immintrin.h>
#include <cstdlib>
#include "voltage.hpp"
using namespace std;

int main(){
int size = 8;
   float ap[] = {1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8, 1.9, 2.0, 2.1, 2.2}; 
   float bp[] = {1.1,1.2,1.3, 1.4, 1.5, 1.6, 1.7, 1.8}; 
   float s = 1.5;
    //vv_dot_product_256(ap,bp,size);
    //vs_multiply(ap, 2, size);
    //mv_multiply_256(ap, bp, 4, 3, 4);
    mm_multiply_256(ap,bp, 6, 2, 2, 4);
    return 0;
}