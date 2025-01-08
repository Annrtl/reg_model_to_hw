#include <json/json.h>
#include "dense.h"

class Parser {
    public:
        Parser(Json::Value model);
        unsigned int getInputShape();
    private:
        Json::Value model;
        unsigned int layersNumber;
        Dense* layers;
        void checkModel();
        void parseModel();
        void parseLayers();
        void parseLayersNumber();
};