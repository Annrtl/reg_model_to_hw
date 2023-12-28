#include "import.h"

unsigned int getInputShape(Json::Value layer){
    if (layer.isMember("build_config")){
        if (layer["build_config"].isMember("input_shape")){
            unsigned int inputShapeSize = layer["build_config"]["input_shape"].size();
            if (inputShapeSize != 2){
                printf("Input shape size not supported: %d\n", inputShapeSize);
            }
            return layer["build_config"]["input_shape"][1].asInt();
        }
    }
    std::cout << "Input shape not supported !" << std::endl;
    return 0;
}

dense_weights_t getWeights(Json::Value json_weights){
    dense_weights_t weights;
    for (int i=0; i<json_weights[0].size(); i++){
        weights.kernel.push_back(json_weights[0][i][0].asFloat());
    }
    weights.bias = json_weights[1][0].asFloat();
    return weights;
}