#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
#include <cstdlib>

using namespace std;

class Matrix{
    private:
        vector<vector<double>> matrix;
        int rows;
        int cols;

    public:
        Matrix(const int& rows, const int& cols){
            this->rows = rows;
            this->cols = cols;
            this->matrix = vector<vector<double>>(rows, vector<double>(cols, 0));
        }

        Matrix(const vector<vector<double>>& matrix){
            this->rows = matrix.size();
            this->cols = matrix[0].size();
            this->matrix = matrix;
        }

        Matrix(const Matrix& m){
            this->rows = m.rows;
            this->cols = m.cols;
            this->matrix = m.matrix;
        }

        void printMatrix(){
            for(int i = 0; i < this->rows; i++){
                for(int j = 0; j < this->cols; j++){
                    cout << this->matrix[i][j] << " ";
                }
                cout << endl;
            }
        }

        void setMatrix(const vector<vector<double>>& matrix){
            this->matrix = matrix;
        }

        vector<vector<double>> getMatrix() const {
            return this->matrix;
        }

        void setElement(const int& row, const int& col, const double& value){
            this->matrix[row][col] = value;
        }

        double getElement(const int& row, const int& col) const {
            return this->matrix[row][col];
        }

        void add(const Matrix& m){
            if(this->rows != m.rows || this->cols != m.cols){
                cout << "Error: matrices must have the same dimensions" << endl;
                return;
            }
            for(int i = 0; i < this->rows; i++){
                for(int j = 0; j < this->cols; j++){
                    this->matrix[i][j] += m.matrix[i][j];
                }
            }
        }

        void subtract(const Matrix& m){
            if(this->rows != m.rows || this->cols != m.cols){
                cout << "Error: matrices must have the same dimensions" << endl;
                return;
            }
            for(int i = 0; i < this->rows; i++){
                for(int j = 0; j < this->cols; j++){
                    this->matrix[i][j] -= m.matrix[i][j];
                }
            }
        }

        Matrix dot(const Matrix& m){
            if(this->cols != m.rows){
                cout << "Error: number of columns of the first matrix must be equal to the number of rows of the second matrix" << endl;
                return Matrix(0, 0);
            }
            Matrix result(this->rows, m.cols);
            for(int i = 0; i < this->rows; i++){
                for(int j = 0; j < m.cols; j++){
                    double sum = 0;
                    for(int k = 0; k < this->cols; k++){
                        sum += this->matrix[i][k] * m.matrix[k][j];
                    }
                    result.matrix[i][j] = sum;
                }
            }
            return result;
        }

        void fillInRange(const double& bottom, const double& top) {
            for(int i = 0; i < this->rows; i++) {
                for(int j = 0; j < this->cols; j++) {
                    double random = ((double)rand() / RAND_MAX) * (top - bottom) + bottom;
                    this->matrix[i][j] = random;
                }
            }
        }
};

int main() {
    // Seed the random number generator
    srand(time(nullptr));
    
    // Create first matrix (2x2) with numbers 1-4
    vector<vector<double>> data1 = {
        {1.0, 2.0},
        {3.0, 4.0}
    };
    Matrix m1(data1);
    
    // Create second matrix (2x1) with random numbers between 1-4
    Matrix m2(2, 1);
    for(int i = 0; i < 2; i++) {
        double random_value = (rand() % 4) + 1;  // Random number between 1-4
        m2.setElement(i, 0, random_value);
    }
    
    // Print the matrices
    cout << "Matrix 1 (2x2):" << endl;
    m1.printMatrix();
    
    cout << "\nMatrix 2 (2x1):" << endl;
    m2.printMatrix();
    
    // Perform dot product
    cout << "\nResult of dot product:" << endl;
    Matrix result = m1.dot(m2);
    result.printMatrix();

    srand(time(nullptr));
    Matrix test(5, 5);
    test.fillInRange(0, 1);
    test.printMatrix();
    
    return 0;
}


