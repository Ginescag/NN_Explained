#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

//a Perceptron is the most basic form of a neural network. It takes in a set of inputs, multiplies them by a set of weights, and then adds them together. 
//If the sum is greater than a certain threshold, the perceptron outputs 1, otherwise it outputs 0. 
//The perceptron can be trained to recognize patterns in data by adjusting the weights. The perceptron is the building block of more complex neural networks.

class Perceptron {
    private:
        double bias;
        double weights[];
};
