#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
#include <cstdlib>

using namespace std;

//a Perceptron is the most basic form of a neural network. It takes in a set of inputs, multiplies them by a set of weights, and then adds them together. 
//If the sum is greater than a certain threshold, the perceptron outputs 1, otherwise it outputs 0. 
//The perceptron can be trained to recognize patterns in data by adjusting the weights. The perceptron is the building block of more complex neural networks.

// int sum(vector<int> inputs, vector<int> weights) {
//     int sum = 0;
//     for (int i = 0; i < inputs.size(); i++) {
//         sum += inputs[i] * weights[i];
//     }
//     return sum;
// }


class Perceptron {
    private:
        double bias;
        vector<double> weights;
    public:
        Perceptron(){
            this->bias = 0;
            for(int i = 0; i < 2; i++){
                this->weights.push_back(2.0 * rand() / RAND_MAX - 1.0);
            }
        }

        Perceptron(const double& bias, const vector<double> weights) {
            this->bias = bias;
            this->weights = weights;
        }

        Perceptron(const Perceptron &p){
            this->bias = p.bias;
            this->weights = p.weights;
        }

        int guess(const vector<double> inputs){
            double sum = 0;
            for(int i = 0; i < weights.size(); i++){
                sum += inputs[i] * this->weights[i];
            }
            if(sum > this->bias){
                return 1;
            } else {
                return -1;
            }
        }

        //aux functions
        void setBias(const double& bias){
            this->bias = bias;
        }

        void setWeights(const vector<double>& weights){
            this->weights = weights;
        }

        double getBias() const {
            return this->bias;
        }

        vector<double> getWeights() const {
            return this->weights;
        }

        void printP() const {
            cout << "Bias: " << this->bias << endl;
            cout << "Weights: ";
            for(int i = 0; i < this->weights.size(); i++){
                cout << this->weights[i] << " ";
            }
            cout << endl;
        }
};

int main() {
    srand(static_cast<unsigned>(time(0)));
    Perceptron p;
    p.printP();
    vector<double> inputs = {0.5, 3};
    cout << p.guess(inputs) << endl;
    return 0;
}
