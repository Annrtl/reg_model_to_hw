#include <vector>

using std::vector;

class Dense {
    public:
        Dense(unsigned int inputShape);
        void setWeights(vector<float> kernel, float bias);
        float infer(vector<float> input);
    private:
        unsigned int inputShape;
        vector<float> kernel;
        float bias;
};
