#include <iostream>
#include <stdio.h>
#include <json/json.h>
#include <fstream>

#include "dense.h"
#include "import.h"
#include "model.h5.h"
#include "model.json.h"

#include <h5cpp/hdf5.hpp>

using namespace hdf5;

void groupDicovering(node::Group grp){
    for(auto link: grp.links)
    {
        if (grp.has_group(link.path())){
            std::cout << "Found group: " << link.path() << std::endl;
            node::Group subGrp = grp.get_group(link.path());
            groupDicovering(subGrp);
        }
        if (grp.has_dataset(link.path())){
            std::cout << "Found dataset: " << link.path() << std::endl;
            node::Dataset dts = grp.get_dataset(link.path());
            std::cout << "Found data: " << dts.datatype().get_class() << std::endl;
            std::cout << "Found data: " << dts.datatype().size() << std::endl;
            std::cout << "Found data: " << dts.dataspace().size() << std::endl;
            for(auto attr: dts.attributes)
            {
                std::cout << "Found attribute: " << attr.name() << std::endl;
            }
        }
    }
    for(auto attr: grp.attributes)
    {
        std::cout << "Found attribute: " << attr.name() << std::endl;
    }
}

int main(){
    Json::Reader reader;
    Json::Value model;
    char json_bin[json_model_bin_len];
    for (int i = 0; i < json_model_bin_len; ++i)
        json_bin[i] = static_cast<char>(json_model_bin[i]);
    char * jsonStart = json_bin;
    char * jsonStop = json_bin + json_model_bin_len;
    reader.parse(jsonStart, jsonStop, model, false);

    std::cout << model << std::endl;

    if (model["class_name"] != "Sequential"){
        printf("Sequential class name not found !\n");
        return 1;
    }

    Json::Value layers = model["config"]["layers"];

    if (layers.size() == 0){
        printf("No layer found !\n");
        return 1;
    }

    unsigned int inputShape;

    for (int i=0; i<layers.size(); i++){
        printf("Analysing layer n° %d/%d\n", i+1, layers.size());

        Json::Value layer = layers[i];

        if (layer["class_name"] != "Dense"){
            std::cout << " - Ignoring layer type: " << layer["class_name"] << std::endl;
            continue;
        }

        std::cout << " - Processing layer type: " << layer["class_name"] << std::endl;

        inputShape = getInputShape(layer);

        std::cout << " - Input shape is: " << inputShape << std::endl;

        unsigned int units = layer["config"]["units"].asInt();
    }

    Dense myLayer = Dense(inputShape);

    //fs::path file_path("../model/model.h5");
    /*
    boost::filesystem::path file_path = "../model/model.h5";
    if(!file::is_hdf5_file(file_path))
    {
        std::cout<<"Error ! File is not an hdf5 file !"<<std::endl;
        return 1;
    }

    file::File f = file::open(file_path);
    node::Group root = f.root();
    */

    node::Group root;
    root.write(ArrayAdapter<unsigned char>(h5_model_bin, h5_model_bin_len));

    groupDicovering(root);

    node::Dataset k_dts = root.get_dataset("/dense/dense/kernel:0");
    node::Dataset b_dts = root.get_dataset("/dense/dense/bias:0");
    std::vector<float> kernel(k_dts.dataspace().size());
    std::vector<float> bias(b_dts.dataspace().size());
    k_dts.read(kernel);
    b_dts.read(bias);
    for (auto k: kernel)
        std::cout << "Value of kernel: " << k << std::endl;
    for (auto b: bias)
        std::cout << "Value of bias: " << b << std::endl;

    myLayer.setWeights(kernel, bias);

    std::vector<float> input = {1, 2, 3, 4, 5, 6, 7, 8};
    std::cout << myLayer.infer(input) << std::endl;

    return 0;
}