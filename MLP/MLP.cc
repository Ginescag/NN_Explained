#include <iostream>
#include <cmath>
#include <vector>
#include <ctime>
#include <cstdlib>

using namespace std;

//a Multilayer Perceptron is a type of feedforward artificial neural network that 
//consists of at least three layers of nodes: an input layer, a hidden layer and an output layer.
//it works by training the network to minimize the error between the actual output and the desired output
//based on perceptrons, which are the basic units of a neural network
//basic perceptrons cannot solve non-linear problems, so the MLP is used to solve non-linear problems
//XOR is a simple example to demonstrate the working of the MLP, as it is a non-linear function

//the MLP class is the main class that contains the functions to train the network and predict the output


class MLP
{
    private:
        double learning_rate; //learning rate is the step size at which the weights are updated
        double bias; //bias is the constant value that is added to the weighted sum of the inputs
        int input_nodes; //number of input nodes
        int hidden_nodes; //number of hidden nodes
        int output_nodes; //number of output nodes

};
