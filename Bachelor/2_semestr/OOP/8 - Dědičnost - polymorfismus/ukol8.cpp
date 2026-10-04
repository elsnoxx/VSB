#include <iostream>
#include <string>
#include <vector>

#include "Task1.h"
#include "GeometricObject.h"

//Úkoly na cvičení
//• Implementuje příklady z přednášky, zaměřte se na využití virtuální metody a pochopení, jak funguje při polymorfním přiřazení.
// 
//• Navrhněte a implementuje jednoduchou dědičnou hierarchii geometrických objektů, které budou mít společné virtuální
//  metody „Obsah“ a „Obvod“.Využijte polymorfní datovou strukturu(např.pole ukazatelů) a rozeberte chování při využití
//  substitučního principu(zejména při srovnání s obyčejným překrytím).

using namespace std;




int main() {
    Client* o = new Client("Jan Novak", "Kralovska 10", "123456789");
    CreditAccount* ca = new CreditAccount(1, o, 1000);
    Account* a = ca;

    cout << "Account balance: " << a->GetBalance() << endl;
    cout << "Account interest rate: " << a->GetInterestRate() << endl;
    cout << "Account owner: " << a->GetOwner()->GetName() << endl;
    cout << "Account owner address: " << a->GetOwner()->GetAddress() << endl;
    cout << "Account owner phone: " << a->GetOwner()->GetPhone() << endl;
    cout << "Account type: " << a->GetAccountType() << endl; // Volání nové virtuální metody

    a->Deposit(500);
    cout << "After depositing 500, balance: " << a->GetBalance() << endl;

    if (a->Withdraw(200)) {
        cout << "Successfully withdrew 200, new balance: " << a->GetBalance() << endl;
    }
    else {
        cout << "Failed to withdraw 200, balance: " << a->GetBalance() << endl;
    }

    if (a->Withdraw(2000)) {
        cout << "Successfully withdrew 2000, new balance: " << a->GetBalance() << endl;
    }
    else {
        cout << "Failed to withdraw 2000, balance: " << a->GetBalance() << endl;
    }

    delete ca;

    cout << endl;
    cout << endl;
    cout << endl;

    std::vector<GeometricObject*> shapes;

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
