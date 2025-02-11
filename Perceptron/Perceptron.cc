#include <GL/glut.h>
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

// Global variables for visualization
vector<vector<double>> points;
vector<int> labels;
vector<bool> correct_guesses;
    Perceptron perceptron;
int current_iteration = 0;
const int TRAINING_POINTS = 100;
const int ITERATIONS = 100;

void display() {
    glClear(GL_COLOR_BUFFER_BIT);
    
    // Draw coordinate axes
    glColor3f(0.5, 0.5, 0.5);
    glBegin(GL_LINES);
        glVertex2f(-1.0, 0.0);
        glVertex2f(1.0, 0.0);
        glVertex2f(0.0, -1.0);
        glVertex2f(0.0, 1.0);
    glEnd();

    // Draw target line (y = 2x)
    glColor3f(0.0, 0.0, 0.0);
    glBegin(GL_LINES);
        glVertex2f(-1.0, -2.0);
        glVertex2f(1.0, 2.0);
    glEnd();

    // Draw points
    glPointSize(5.0);
    for(size_t i = 0; i < points.size(); i++) {
        if(current_iteration > 0) {
            if(correct_guesses[i]) {
                glColor3f(0.0, 1.0, 0.0);  // Green for correct
            } else {
                glColor3f(1.0, 0.0, 0.0);  // Red for incorrect
            }
        } else {
            glColor3f(0.0, 0.0, 0.0);  // Black for initial state
        }

        glBegin(GL_POINTS);
            glVertex2f(points[i][0]/3.0, points[i][1]/3.0);  // Scale down to fit in [-1,1]
        glEnd();
    }

    glutSwapBuffers();
}

void keyboard(unsigned char key, int x, int y) {
    if(key == ' ' && current_iteration < ITERATIONS) {
        // Train one iteration
        int correct_predictions = 0;
        for(int j = 0; j < TRAINING_POINTS; j++) {
            int guess = perceptron.guess(points[j]);
            correct_guesses[j] = (guess == labels[j]);
            if(correct_guesses[j]) correct_predictions++;
            perceptron.train(points[j], labels[j]);
        }

        double accuracy = (correct_predictions * 100.0) / TRAINING_POINTS;
        cout << "Iteration " << current_iteration + 1 << ": " << accuracy << "%" << endl;
        perceptron.printP();
        current_iteration++;
        
        glutPostRedisplay();
    }
}

int main(int argc, char** argv) {
    srand(static_cast<unsigned>(time(0)));
    
    // Generate training data
    correct_guesses.resize(TRAINING_POINTS);
    for(int i = 0; i < TRAINING_POINTS; i++) {
        double x = (rand() / double(RAND_MAX)) * 6 - 3;
        double y = (rand() / double(RAND_MAX)) * 6 - 3;
        points.push_back({x, y});
        labels.push_back(y > 2 * x ? 1 : -1);
    }

    // Initialize GLUT
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(800, 800);
    glutCreateWindow("Perceptron Learning");

    // Set white background
    glClearColor(1.0, 1.0, 1.0, 1.0);

    // Register callbacks
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);

    // Start main loop
    glutMainLoop();
    return 0;
}
