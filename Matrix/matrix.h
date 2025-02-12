#ifndef _MATRIX_H_
#define _MATRIX_H_

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
        Matrix(const int& rows, const int& cols);

        Matrix(const vector<vector<double>>& matrix);

        Matrix(const Matrix& m);

        void printMatrix();

        void setMatrix(const vector<vector<double>>& matrix);

        vector<vector<double>> getMatrix() const;

        void setElement(const int& row, const int& col, const double& value);

        double getElement(const int& row, const int& col) const;

        void add(const Matrix& m);

        void subtract(const Matrix& m);

        Matrix dot(const Matrix& m);

        void fillInRange(const double& bottom, const double& top);
};

#endif
























