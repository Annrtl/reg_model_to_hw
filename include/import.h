#include <json/json.h>
#include <iostream>
#include <vector>

using std::vector;

struct dense_weights_t {
    vector<float> kernel;
    float bias;
};

unsigned int getInputShape(Json::Value layer);
dense_weights_t getWeights(Json::Value json_weights);