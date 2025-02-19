#include <iostream>
#include <cmath>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <iomanip>
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


double sigmoid(double x){
    return 1/(1+exp(-x));
}

double dsigmoid(double y){
    return y * (1 - y);
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
        
        
        vector<Matrix> feedforward(vector<double> input) {
            Matrix layer_input = Matrix(input);
            Matrix result = layer_input;
            vector<Matrix> results;
            
            //process each layer 1 by 1
            for(int i = 0; i < weights.size(); i++) {
                result = processLayer(result, weights[i], bias[i]);
                results.push_back(result);
            }
            
            return results;
        }

        void backpropagation(const vector<double>& input, const vector<double>& expectedO){
            //this is a bit confusing as im taking arrays as parameters to convert them into matrices
            //however this will make testing much easier to visualize and understand

            vector<Matrix> Results = feedforward(input);
            Matrix output = Results[Results.size()-1];
            Results.pop_back();
            vector<Matrix> hiddenResults = Results;
            Matrix expectedOutput = Matrix(expectedO);
            
            //calculate the  output error
            Matrix OutputErrors = expectedOutput.subtract(output);
            vector<Matrix> hiddenErrors;
            
            //calculate the hidden errors
            for(int i = weights.size()-1; i >= 0; i--){
                if(i == weights.size()-1){
                    Matrix error = weights[i].transpose().dot(OutputErrors);
                    hiddenErrors.insert(hiddenErrors.begin(), error);
                } else {
                    Matrix error = weights[i].transpose().dot(hiddenErrors.front());
                    hiddenErrors.insert(hiddenErrors.begin(), error);
                }
            }
            //calculate deltas
            Matrix outputGradient = Matrix::mapStatic(dsigmoid, output);
            outputGradient.hadamard(OutputErrors);
            outputGradient.scalarMultiply(learning_rate);
            
            Matrix hiddenT = hiddenResults[hiddenResults.size()-1].transpose();
            Matrix deltaOutput = outputGradient.dot(hiddenT);
            weights[weights.size()-1].add(deltaOutput);
            bias[weights.size()-1].add(outputGradient);


            for(int i = hiddenResults.size()-1; i >= 0; i--){
                Matrix hiddenGradient = Matrix::mapStatic(dsigmoid, hiddenResults[i]);
                hiddenGradient.hadamard(hiddenErrors[i]);
                hiddenGradient.scalarMultiply(learning_rate);

                Matrix inputT(1, 1); // Initialize with temporary dimensions
                if(i == 0){
                    inputT = Matrix(input).transpose();
                } else {
                    inputT = hiddenResults[i-1].transpose();
                }
                Matrix deltaHidden = hiddenGradient.dot(inputT);
                weights[i].add(deltaHidden);;
                bias[i].add(hiddenGradient);
                
            }
        }
};

void printVector(const std::vector<double>& vec, const std::string& label) {
    std::cout << label << ": [";
    for (const auto& val : vec) {
        std::cout << std::fixed << std::setprecision(4) << val << " ";
    }
    std::cout << "]" << std::endl;
}

void printMatrix(const Matrix& m, const std::string& label) {
    std::cout << label << ":" << std::endl;
    m.printMatrix();
    std::cout << std::endl;
}

int main() {
    // Create a simple MLP for XOR
    // 2 inputs -> 4 neurons hidden layer -> 4 neurons hidden layer -> 1 output
    MLP mlp(0.1, 1.0, 2, {4, 4}, 1, sigmoid);
    
    // Test data (XOR example: 1 XOR 0 = 1)
    std::vector<double> input = {1.0, 0.0};
    std::vector<double> expected = {1.0};

    std::cout << "=== Testing Neural Network ===" << std::endl;
    std::cout << "\nInput values:" << std::endl;
    printVector(input, "Input");
    printVector(expected, "Expected Output");

    // Initial output before training
    std::cout << "\n=== Initial Output ===" << std::endl;
    vector<Matrix> initial_results = mlp.feedforward(input);
    std::cout << "Output before training:" << std::endl;
    initial_results[initial_results.size()-1].printMatrix();

    // Training process
    std::cout << "\n=== Training Process ===" << std::endl;
    for(int i = 0; i < 50; i++) {
        std::cout << "\nIteration " << i + 1 << ":" << std::endl;
        std::cout << "----------------------" << std::endl;
        
        // Perform backpropagation
        mlp.backpropagation(input, expected);
        
        // Get current output
        vector<Matrix> current_results = mlp.feedforward(input);
        std::cout << "Current output:" << std::endl;
        current_results[current_results.size()-1].printMatrix();
        
        // Calculate and display error
        double error = abs(expected[0] - current_results[current_results.size()-1].getElement(0, 0));
        std::cout << "Error: " << std::fixed << std::setprecision(6) << error << std::endl;
    }

    // Show final results
    std::cout << "\n=== Final Results ===" << std::endl;
    vector<Matrix> final_results = mlp.feedforward(input);
    std::cout << "Final output:" << std::endl;
    final_results[final_results.size()-1].printMatrix();
    
    double final_error = abs(expected[0] - final_results[final_results.size()-1].getElement(0, 0));
    std::cout << "Final error: " << std::fixed << std::setprecision(6) << final_error << std::endl;

    return 0;
}
