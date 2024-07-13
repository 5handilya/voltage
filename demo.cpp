#include <stdio.h>
#include <iostream>
#include <immintrin.h>
#include <cstdlib>
#include "voltage.hpp"
using namespace std;

int main(){
    int size = 11;
    float* ap = (float*) aligned_alloc(32, sizeof(float)*size);  
    float* bp = (float*) aligned_alloc(32, sizeof(float)*size);  
    if(ap == nullptr || bp == nullptr){
        cout << "failed to allocate aligned memory" << endl;
    }
    else{
        float values[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.5f, 11.2f};
        for (int i = 0; i < size; i++) {
        ap[i] = values[i];
        bp[i] = values[i];
        }
    }
    vv_dot_product_256(ap,bp,size);
    free(ap);
    free(bp);
    return 0;
}