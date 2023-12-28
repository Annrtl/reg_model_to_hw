#include <iostream>
#include <stdio.h>
#include <json/json.h>
#include <fstream>

#include "dense.h"
#include "import.h"
#include "model.json.h"

using std::cout;
using std::endl;

int main(int argc, char *argv[]){
    Json::Reader reader;
    Json::Value model;
    char json_bin[json_model_bin_len];
    for (int i = 0; i < json_model_bin_len; ++i)
        json_bin[i] = static_cast<char>(json_model_bin[i]);
    char * jsonStart = json_bin;
    char * jsonStop = json_bin + json_model_bin_len;
    reader.parse(jsonStart, jsonStop, model, false);

    //cout << model << endl;

    if (model["class_name"] != "Sequential"){
        printf("Sequential class name not found !\n");
        return 1;
    }

    Json::Value layers = model["config"]["layers"];

    if (layers.size() == 0){
        //printf("No layer found !\n");
        return 1;
    }

    unsigned int inputShape;

    for (int i=0; i<layers.size(); i++){
        //printf("Analysing layer n° %d/%d\n", i+1, layers.size());

        Json::Value layer = layers[i];

        if (layer["class_name"] != "Dense"){
            //cout << " - Ignoring layer type: " << layer["class_name"] << endl;
            continue;
        }

        //cout << " - Processing layer type: " << layer["class_name"] << endl;

        inputShape = getInputShape(layer);

        //cout << " - Input shape is: " << inputShape << endl;

        unsigned int units = layer["config"]["units"].asInt();
    }

    Dense myLayer = Dense(inputShape);
    dense_weights_t weights = getWeights(model["weights"]);
    myLayer.setWeights(weights.kernel, weights.bias);

    if (argc <= inputShape){
        cout << "[ERROR] Input shape must be: " << inputShape << endl;
        cout << "[INFO] Input shape is: " << argc-1 << endl;
        return 1;
    }

    vector<float> input;

    for (int i=1; i<argc; i++){
        float val = atof(argv[i]);
        //cout << val << endl;
        input.push_back(val);
    }

    cout << myLayer.infer(input) << endl;
    return 0;
}