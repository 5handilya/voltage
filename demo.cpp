#include <stdio.h>
#include <iostream>
#include <immintrin.h>
#include <cstdlib>
#include "voltage.hpp"
using namespace std;

int main(){
    int size = 11;
   float ap[] = {1.0, 1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0}; 
   float bp[] = {1.0, 1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0}; 
    vv_dot_product_256(ap,bp,size);
    
    
    return 0;
}