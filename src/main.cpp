#include <iostream>
#include <stdio.h>
#include <json/json.h>
#include <fstream>

#include "import.h"
#include "parser.h"
#include "model.json.h"

using std::cout;
using std::endl;

int main(int argc, char *argv[]){
    // Load the pseudo-json (header file)
    Json::Reader reader;
    Json::Value model;
    char json_bin[json_model_bin_len];
    for (int i = 0; i < json_model_bin_len; ++i)
        json_bin[i] = static_cast<char>(json_model_bin[i]);
    char * jsonStart = json_bin;
    char * jsonStop = json_bin + json_model_bin_len;
    reader.parse(jsonStart, jsonStop, model, false);

    // Create the JSON parser
    Parser parser = Parser(model);

    // Check input Shape integrity
    if (argc <= parser.getInputShape()){
        cout << "[ERROR] Input shape must be: " << parser.getInputShape() << endl;
        cout << "[INFO] Input shape is: " << argc-1 << endl;
        return 1;
    }

    // Convert input to fill input vector
    vector<float> input;

    for (int i=1; i<argc; i++){
        float val = atof(argv[i]);
        //cout << val << endl;
        input.push_back(val);
    }

    return 0;
}