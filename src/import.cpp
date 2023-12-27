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