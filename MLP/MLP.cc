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
//to simplify the implementation, this model will only have 2 hidden layers and only use the sigmoid activation function


int sigmoid(double x){
    return 1/(1+exp(-x));
}
class MLP
{
    private:
        double learning_rate; //learning rate is the step size at which the weights are updated
        vector<Matrix> bias; //bias is the constant value that is added to the weighted sum of the inputs
        int input_nodes; //number of input nodes
        vector<int> hidden_layers; //vector with the number of nodes in each hidden layer
        int output_nodes; //number of output nodes
        double (*activation_function)(double);
        vector<Matrix> weights; //vector of matrices that contain the weights of the network, first matrix is the weights between the input layer and the first hidden layer, last matrix is the weights between the last hidden layer and the output layer


        Matrix processLayer(const Matrix& input, const Matrix& weights, const Matrix& bias) {
            Matrix result = weights.dot(input);
            result.add(bias);
            result.map(activation_function);
            return result;
        }
    public:

        //this needs some more logic as pushing back a new hidden layer will fuck up the weigths vector (not implementing it anytime soon)
        void addLayer(int nodes){
            hidden_layers.push_back(nodes);
        }

        MLP(double lr, double b, int in, vector<int> h, double out, double (*activation)(double)){
            learning_rate = lr;
            input_nodes = in;
            hidden_layers = h;
            output_nodes = out;
            activation_function = activation;
            
            //b is a vector with the biases of each hidden layer, the last one is the bias for the output layer
            
            //add weights matrices
            this->weights.push_back(Matrix(hidden_layers[0], input_nodes));

            for(int i = 1; i < hidden_layers.size(); i++){
                this->weights.push_back(Matrix(hidden_layers[i], hidden_layers[i-1]));
            }

            this->weights.push_back(Matrix(output_nodes, hidden_layers[hidden_layers.size()-1]));

            // fill those matrices with random numbers
            for(int i = 0; i < weights.size(); i++){
                weights[i].randomize();
            }

            //add bias matrices
            for(int i = 0; i < hidden_layers.size(); i++){
                bias.push_back(Matrix(hidden_layers[i], 1));
                bias[i].randomize();
            }

            bias.push_back(Matrix(output_nodes, 1));
            bias[hidden_layers.size()].randomize();
        }
        
        
        vector<double> feedforward(vector<double> input) {
            Matrix layer_input = Matrix(input);
            Matrix result = layer_input;
            
            //process each layer 1 by 1
            for(int i = 0; i < weights.size(); i++) {
                result = processLayer(result, weights[i], bias[i]);
            }
            
            return result.toVector();
        }
        
};
