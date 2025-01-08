#include "parser.h"

Parser::Parser(Json::Value model){
    this->model = model;
    this->checkModel();
    this->parseModel();
}

void Parser::checkModel(){
    // Check the model type is Sequential;
    if (this->model["class_name"] != "Sequential"){
        printf("Only sequential model supported for now !\n");
    }

    // Check at least one layer is defined
    if (this->model["config"]["layers"].size() == 0){
        printf("No layer found !\n");
    }
}

unsigned int Parser::getInputShape(){
    for (int i=0; i<this->layersNumber; i++){
        if (this->model['config']['layers'][i]['class_name'] != "InputLayer")
            continue;
        return this->model['config']['layers'][i]['class_name']['batch_input_shape'][1].asInt();
    }
    return 0;
}

void Parser::parseLayersNumber(){
    this->layersNumber = this->model['config']['layers'].size();
}

void Parser::parseLayers(){
    this->parseLayersNumber();
    this->layers = (Dense*)malloc(sizeof(Dense) * this->layersNumber);
    for (int i=0; i<this->layersNumber; i++){
        Json::Value layer = layers[i];
        if (layer["class_name"] == "InputLayer"){
            inputShape = getInputShape(layer);
        } else if (layer["class_name"] == "Dense"){
            inputShape = getInputShape(layer);
            inputShape = layer["config"]["units"].asInt();
        } else {
            printf("Layer n° %d/%d is ignored (unknown type)\n", i+1, this->layersNumber);
        }
        c_layers[i] = Dense(inputShape);
        dense_weights_t weights = getWeights(model["weights"]);
        c_layers[i].setWeights(weights.kernel, weights.bias);
    }
}

void Parser::parseModel(){
    this->parseLayers();
}

unsigned int Parser::getLayersNumber(){
    return this->layers.size()
}

Dense* Parser::getLayers(){
    return this->layers;
}
