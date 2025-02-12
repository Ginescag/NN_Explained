#include <iostream>
#include <cmath>
#include <vector>
#include <ctime>
#include <cstdlib>
#include "../Matrix/matrix.h"

using namespace std;

//g++ -g -Iinclude MLP/MLP.cc Matrix/matrix.o -o MLP


//a Multilayer Perceptron is a type of feedforward artificial neural network that 
//consists of at least three layers of nodes: an input layer, a hidden layer and an output layer.
//it works by training the network to minimize the error between the actual output and the desired output
//based on perceptrons, which are the basic units of a neural network
//basic perceptrons cannot solve non-linear problems, so the MLP is used to solve non-linear problems
//XOR is a simple example to demonstrate the working of the MLP, as it is a non-linear function

//the MLP class is the main class that contains the functions to train the network and predict the output

int sigmoid(double x){
    return 1/(1+exp(-x));
}
class MLP
{
    private:
        double learning_rate; //learning rate is the step size at which the weights are updated
        double bias; //bias is the constant value that is added to the weighted sum of the inputs
        int input_nodes; //number of input nodes
        int hidden_layers; //number of hidden layers
        int nodes_per_hidden_layer; //number of nodes per hidden layer
        int output_nodes; //number of output nodes



    public:
        MLP(const int& inputN, const int& NhiddenL, const int& outputN, const int& hiddenN){
            this -> input_nodes = inputN;
            this -> nodes_per_hidden_layer = hiddenN;
            this -> hidden_layers = NhiddenL;
            this -> output_nodes = outputN;
        }

};
