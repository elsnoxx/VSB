#include <iostream>
#include <string>
#include <vector>

#include "Task1.h"
#include "GeometricObject.h"

//Úkoly na cvičení
//• Implementuje příklady z přednášky, zaměřte se na využití abstraktní a čistě abstraktní třídy a na to, jak fungují konstruktory a destruktory.
 

//• Reimplementuje jednoduchou dědičnou hierarchii geometrických objektů tak, aby jste využili abstraktní a čistě abstraktní třídy.

using namespace std;

int main() {
    Client* o = new Client("Jan Novak", "Karlovo namesti 13", "123456789");
    CreditAccount* ca = new CreditAccount(1, o, 0.1);

    AbstractAccount* aa = ca;

    delete aa;
    delete o;

    getchar();
    std::cout << endl;
    std::cout << endl;

    std::vector<GeometricObjectAbstract*> shapes;

    shapes.push_back(new Circle(5));
    shapes.push_back(new Rectangle(4, 6));
    shapes.push_back(new Square(4));
    shapes.push_back(new Triangle(3, 4, 5));

    for (const auto& shape : shapes) {
        std::cout << "Obvod: " << shape->obvod() << ", Obsah: " << shape->obsah() << std::endl;
    }

    for (auto& shape : shapes) {
        delete shape;
    }
    return 0;
}