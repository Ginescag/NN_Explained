#include <iostream>
#include <cassert>
#include <cmath>
#include "matrix.h"

double testFunction(double x) {
    return x * 2;
}

void printTestHeader(const std::string& testName) {
    std::cout << "\n=== Testing " << testName << " ===" << std::endl;
}

void printOperationHeader(const std::string& operation) {
    std::cout << "\n--- " << operation << " ---" << std::endl;
}

bool areEqual(double a, double b, double epsilon = 0.0001) {
    return std::abs(a - b) < epsilon;
}

int main() {
    // Test constructors
    printTestHeader("Constructors");
    
    // Test basic constructor
    Matrix m1(2, 3);
    std::cout << "Basic constructor (2x3 matrix):" << std::endl;
    m1.printMatrix();

    // Test vector<vector<double>> constructor
    std::vector<std::vector<double>> testData = {{1, 2, 3}, {4, 5, 6}};
    Matrix m2(testData);
    std::cout << "\nVector constructor matrix:" << std::endl;
    m2.printMatrix();

    // Test arithmetic operations
    printTestHeader("Arithmetic Operations");
    
    // Test addition
    printOperationHeader("Matrix Addition");
    Matrix addTest1({{1, 2}, {3, 4}});
    Matrix addTest2({{5, 6}, {7, 8}});
    std::cout << "First matrix:" << std::endl;
    addTest1.printMatrix();
    std::cout << "Second matrix:" << std::endl;
    addTest2.printMatrix();
    addTest1.add(addTest2);
    std::cout << "Result:" << std::endl;
    addTest1.printMatrix();

    // Test scalar addition
    printOperationHeader("Scalar Addition (adding 2)");
    Matrix scalarAdd({{1, 2}, {3, 4}});
    std::cout << "Original matrix:" << std::endl;
    scalarAdd.printMatrix();
    scalarAdd.add(2);
    std::cout << "After adding 2:" << std::endl;
    scalarAdd.printMatrix();

    // Test subtraction
    printOperationHeader("Matrix Subtraction");
    Matrix subTest1({{5, 6}, {7, 8}});
    Matrix subTest2({{1, 2}, {3, 4}});
    std::cout << "First matrix:" << std::endl;
    subTest1.printMatrix();
    std::cout << "Second matrix:" << std::endl;
    subTest2.printMatrix();
    Matrix subResult = subTest1.subtract(subTest2);
    std::cout << "Result:" << std::endl;
    subResult.printMatrix();

    // Test dot product
    printOperationHeader("Dot Product");
    Matrix dotTest1({{1, 2}, {3, 4}});
    Matrix dotTest2({{5, 6}, {7, 8}});
    std::cout << "First matrix:" << std::endl;
    dotTest1.printMatrix();
    std::cout << "Second matrix:" << std::endl;
    dotTest2.printMatrix();
    Matrix dotResult = dotTest1.dot(dotTest2);
    std::cout << "Result:" << std::endl;
    dotResult.printMatrix();

    // Test Hadamard product
    printOperationHeader("Hadamard Product");
    std::cout << "First matrix:" << std::endl;
    dotTest1.printMatrix();
    std::cout << "Second matrix:" << std::endl;
    dotTest2.printMatrix();
    Matrix hadamardResult = dotTest1.hadamard(dotTest2);
    std::cout << "Result:" << std::endl;
    hadamardResult.printMatrix();

    // Test transpose
    printOperationHeader("Transpose");
    Matrix transposeTest({{1, 2, 3}, {4, 5, 6}});
    std::cout << "Original matrix:" << std::endl;
    transposeTest.printMatrix();
    Matrix transposed = transposeTest.transpose();
    std::cout << "Transposed matrix:" << std::endl;
    transposed.printMatrix();

    // Test mapping functions
    printTestHeader("Mapping Functions");
    Matrix mapTest({{1, 2}, {3, 4}});
    std::cout << "Original matrix:" << std::endl;
    mapTest.printMatrix();
    mapTest.map(testFunction);
    std::cout << "After mapping (x2):" << std::endl;
    mapTest.printMatrix();

    // Test static mapping
    printOperationHeader("Static Mapping");
    Matrix staticMapInput({{1, 2}, {3, 4}});
    std::cout << "Input matrix:" << std::endl;
    staticMapInput.printMatrix();
    Matrix staticMapResult = Matrix::mapStatic(testFunction, staticMapInput);
    std::cout << "Result after static mapping (x2):" << std::endl;
    staticMapResult.printMatrix();

    // Test randomization
    printTestHeader("Randomization");
    Matrix randomTest(2, 2);
    randomTest.randomize();
    std::cout << "Randomized matrix (-1 to 1):" << std::endl;
    randomTest.printMatrix();

    // Test fillInRange
    printOperationHeader("Fill in Range (0 to 10)");
    Matrix rangeTest(2, 2);
    rangeTest.fillInRange(0, 10);
    std::cout << "Matrix filled in range 0-10:" << std::endl;
    rangeTest.printMatrix();

    std::cout << "\nAll tests completed!" << std::endl;
    return 0;
}