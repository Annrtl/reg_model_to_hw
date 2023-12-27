#include <vector>

class Dense {
    public:
        Dense(unsigned int inputShape);
        void setWeights(std::vector<float> kernel, std::vector<float> bias);
        float infer(std::vector<float> input);
    private:
        unsigned int inputShape;
        std::vector<float> kernel;
        std::vector<float> bias;
};
