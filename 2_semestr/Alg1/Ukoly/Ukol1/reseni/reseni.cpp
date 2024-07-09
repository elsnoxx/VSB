#include <iostream>
#include <cmath>

double diskriminantCalc(double a, double b, double c) {
    return pow(b, 2) - 4 * a * c;
}

double koren1Calc(double a, double b, double diskriminant) {
    return (-b + sqrt(diskriminant)) / 2 * a;
}

double koren2Calc(double a, double b, double diskriminant) {
    return (-b - sqrt(diskriminant)) / 2 * a;
}


int calculate(double a, double b, double c, double &x1, double& x2) {
    double diskriminant = diskriminantCalc(a, b, c);
    if (diskriminant > 0)
    {
        x1 = koren1Calc(a, b, diskriminant);
        x2 = koren2Calc(a, b, diskriminant);
        return 2;
    }
    else if (diskriminant == 0)
    {
        x1 = koren1Calc(a, b, diskriminant);
        return 1;
    }
    else
    {
        std::cout << "0" << "\n";
        return 0;
    }
}


int main()
{    
    // x^2 − 8𝑥 + 7 = 0 ====> 2 ====> x1 = 7, x2 = 1

    double a = 1;
    double b = -8;
    double c = 7;


    // x^2 + 2𝑥 + 1 = 0 ====> 1 ====> x1 = -1
    /*
    double a = 1;
    double b = 2;
    double c = 1;
    */

    // x^2 + 1 = 0 ====> 0 ====>
    /*
    double a = 1;
    double b = 0;
    double c = 1;
    */

    double x1;
    double x2;
    calculate(a,b,c,x1,x2);
    // std::cout << x1 << "\n";
    // std::cout << x2 << "\n";
}
