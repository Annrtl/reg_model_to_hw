#include "dense.h"

Dense::Dense(unsigned int inputShape){
    this->inputShape = inputShape;
}

float Dense::infer(std::vector<float> input){
    float result = 0;
    for (int i=0; i<this->inputShape; i++){
        result += input[i] * this->kernel[i];
    }
    result += this->bias[0];
    return result;
}

void Dense::setWeights(std::vector<float> kernel, std::vector<float> bias){
    this->kernel = kernel;
    this->bias = bias;
}