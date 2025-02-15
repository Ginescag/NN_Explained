#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include "matrix.h"

using namespace std;

Matrix::Matrix(const int& rows, const int& cols){
    this->rows = rows;
    this->cols = cols;
    this->matrix = vector<vector<double>>(rows, vector<double>(cols, 0));
}

Matrix::Matrix(const vector<vector<double>>& matrix){
    this->rows = matrix.size();
    this->cols = matrix[0].size();
    this->matrix = matrix;
}

Matrix::Matrix(const Matrix& m){
    this->rows = m.rows;
    this->cols = m.cols;
    this->matrix = m.matrix;
}

void Matrix::printMatrix(){
    for(int i = 0; i < this->rows; i++){
        for(int j = 0; j < this->cols; j++){
            cout << this->matrix[i][j] << " ";
        }
        cout << endl;
    }
}

void Matrix::setMatrix(const vector<vector<double>>& matrix){
    this->matrix = matrix;
}

vector<vector<double>> Matrix::getMatrix() const {
    return this->matrix;
}

void Matrix::setElement(const int& row, const int& col, const double& value){
    this->matrix[row][col] = value;
}

double Matrix::getElement(const int& row, const int& col) const {
    return this->matrix[row][col];
}

void Matrix::add(const Matrix& m){
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

void Matrix::addScalar(const double& s){
    for(int i = 0; i < this->rows; i++){
        for(int j = 0; j < this->cols; j++){
            this->matrix[i][j] += s;
        }
    }
}

void Matrix::subtract(const Matrix& m){
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

void Matrix::subtractScalar(const double& s){
    for(int i = 0; i < this->rows; i++){
        for(int j = 0; j < this->cols; j++){
            this->matrix[i][j] -= s;
        }
    }
}

Matrix Matrix::dot(const Matrix& m){
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

Matrix Matrix::hadamard(const Matrix& m){
    if(this->rows != m.rows || this->cols != m.cols){
        cout << "Error: matrices must have the same dimensions" << endl;
        return Matrix(0, 0);
    }
    Matrix result(this->rows, this->cols);
    for(int i = 0; i < this->rows; i++){
        for(int j = 0; j < this->cols; j++){
            result.matrix[i][j] = this->matrix[i][j] * m.matrix[i][j];
        }
    }
    return result;
}

void Matrix::scalarMultiply(const double& s){
    for(int i = 0; i < this->rows; i++){
        for(int j = 0; j < this->cols; j++){
            this->matrix[i][j] *= s;
        }
    }
}

Matrix Matrix::transpose(){
    Matrix result(this->cols, this->rows);
    for(int i = 0; i < this->rows; i++){
        for(int j = 0; j < this->cols; j++){
            result.matrix[j][i] = this->matrix[i][j];
        }
    }
    return result;
}

void Matrix::map(double (*func)(double)){
    for(int i = 0; i < this->rows; i++){
        for(int j = 0; j < this->cols; j++){
            this->matrix[i][j] = func(this->matrix[i][j]);
        }
    }
}

void Matrix::randomize(){
    srand(time(0));
    for(int i = 0; i < this->rows; i++){
        for(int j = 0; j < this->cols; j++){
            this->matrix[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
        }
    }
}

void Matrix::fillInRange(const double& bottom, const double& top) {
    for(int i = 0; i < this->rows; i++) {
        for(int j = 0; j < this->cols; j++) {
            double random = ((double)rand() / RAND_MAX) * (top - bottom) + bottom;
            this->matrix[i][j] = random;
        }
    }
}


