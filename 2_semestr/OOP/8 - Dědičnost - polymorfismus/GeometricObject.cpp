#include "GeometricObject.h"

// Circle methods
Circle::Circle(double r) : radius(r) {}

double Circle::obvod() const {
    return 2 * M_PI * radius;
}

double Circle::obsah() const {
    return M_PI * radius * radius;
}

// Rectangle methods
Rectangle::Rectangle(double w, double h) : width(w), height(h) {}

double Rectangle::obvod() const {
    return 2 * (width + height);
}

double Rectangle::obsah() const {
    return width * height;
}

// Square methods
Square::Square(double s) : side(s) {}

double Square::obvod() const {
    return 4 * side;
}

double Square::obsah() const {
    return side * side;
}

// Triangle methods
Triangle::Triangle(double s1, double s2, double s3) : side1(s1), side2(s2), side3(s3) {}

double Triangle::obvod() const {
    return side1 + side2 + side3;
}

double Triangle::obsah() const {
    double s = (side1 + side2 + side3) / 2;
    return sqrt(s * (s - side1) * (s - side2) * (s - side3));
}
