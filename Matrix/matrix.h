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

        Matrix(const vector<double>& v);

        void printMatrix() const;

        void setMatrix(const vector<vector<double>>& matrix);

        vector<vector<double>> getMatrix() const;

        void setElement(const int& row, const int& col, const double& value);

        double getElement(const int& row, const int& col) const;

        void add(const Matrix& m);

        void add(const double& s);

        Matrix subtract(const Matrix& m) const;

        void subtractScalar(const double& s);

        Matrix dot(const Matrix& m) const;

        void scalarMultiply(const double& s);

        Matrix hadamard(const Matrix& m);

        Matrix transpose();

        void map(double (*func)(double));

        static Matrix mapStatic(double (*func)(double), const Matrix& m);

        void randomize();

        vector<double> toVector() const;

        void fillInRange(const double& bottom, const double& top);
};

#endif
























