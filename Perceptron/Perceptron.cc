#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
#include <cstdlib>

using namespace std;

//a Perceptron is the most basic form of a neural network. It takes in a set of inputs, multiplies them by a set of weights, and then adds them together. 
//If the sum is greater than a certain threshold, the perceptron outputs 1, otherwise it outputs 0. 
//The perceptron can be trained to recognize patterns in data by adjusting the weights. The perceptron is the building block of more complex neural networks.

class Perceptron {
    private:
        double bias;
        vector<double> weights;
        double learning_rate = 0.1;
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

        void train(const vector<double>& inputs, const int& target){
            int guess = this->guess(inputs);
            double error = target - guess;

            //tune all the weights
            for(int i = 0; i < this->weights.size(); i++){
                this->weights[i] += error * inputs[i] * this->learning_rate;
            }
        }
};



int main() {
    srand(static_cast<unsigned>(time(0)));
    Perceptron p;
    
    // Training parameters
    const int TRAINING_POINTS = 250;
    const int ITERATIONS = 100;
    
    // Function to determine if a point is above or below the line y = x
    auto classify_point = [](double x, double y) -> int {
        // Returns 1 if point is above the line y = x, -1 if below
        return y > x ? 1 : -1;
    };

    // Generate training points
    vector<vector<double>> points;
    vector<int> labels;
    
    cout << "Training points:" << endl;
    for(int i = 0; i < TRAINING_POINTS; i++) {
        double x = (rand() / double(RAND_MAX)) * 8 - 4;  // x between -4 and 4
        double y = (rand() / double(RAND_MAX)) * 8 - 4;  // y between -4 and 4
        points.push_back({x, y});
        labels.push_back(classify_point(x, y));
        cout << "Point " << i << ": (" << x << ", " << y << ") -> " 
             << (labels[i] == 1 ? "Above" : "Below") << endl;
    }

    cout << "\nTraining start:" << endl;
    cout << "Initial state:" << endl;
    p.printP();
    cout << "------------------------" << endl;
    
    for(int i = 0; i < ITERATIONS; i++) {
        int correct_predictions = 0;
        
        // Train with all points
        for(int j = 0; j < TRAINING_POINTS; j++) {
            if(p.guess(points[j]) == labels[j]) {
                correct_predictions++;
            }
            p.train(points[j], labels[j]);
        }

        double accuracy = (correct_predictions * 100.0) / TRAINING_POINTS;
        cout << "\nIteration " << i + 1 << ":" << endl;
        cout << "Accuracy: " << accuracy << "%" << endl;
        p.printP();
        cout << "------------------------" << endl;

        if(correct_predictions == TRAINING_POINTS) {
            cout << "\nPerfect accuracy achieved!" << endl;
            break;
        }
    }

    // Test with specific points
    cout << "\nTesting new points:" << endl;
    vector<vector<double>> test_points = {
        {1, 2},    // Should be above line (positive)
        {2, 1},    // Should be below line (negative)
        {-2, -1},  // Should be above line (positive)
        {-1, -2}   // Should be below line (negative)
    };

    for(const auto& point : test_points) {
        int result = p.guess(point);
        cout << "Point (" << point[0] << "," << point[1] << "): " 
             << (result == 1 ? "Above" : "Below") << " y=x" << endl;
    }

    return 0;
}
