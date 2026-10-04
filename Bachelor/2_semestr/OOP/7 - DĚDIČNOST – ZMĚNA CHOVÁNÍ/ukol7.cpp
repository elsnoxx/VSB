#include <iostream>
#include <cmath>
#include <string>

#include "GeometricObject.h"
#include "Task1.h"

//Úkoly na cvičení
// • Implementujte příklady z přednášky.Zaměřte se na překrytí, vyzkoušejte použití „protected“.
//
// • Navrhněte a implementuje jednoduchou dědičnou hierarchii geometrických objektů, které budou mít
//   společné metody „Obsah“ a „Obvod“.Využijte překrytí a rozeberte chování při využití substitučního principu.

using namespace std;

int main() {

    Client* client = new Client("Jan Novak", "Kralovska 10", "123456789");
    Account* account = new Account(1, client, 0.01);
    CreditAccount* creditAccount = new CreditAccount(2, client, 0.01, 1000);

    account->Deposit(1000);
    creditAccount->Deposit(1000);

    cout << "Account balance: " << account->GetBalance() << endl;
    cout << "Credit account balance: " << creditAccount->GetBalance() << endl;

    account->AddInterest();
    creditAccount->AddInterest();

    cout << "Account balance: " << account->GetBalance() << endl;
    cout << "Credit account balance: " << creditAccount->GetBalance() << endl;

    account->Withdraw(500);
    creditAccount->Withdraw(500);

    cout << "Account balance: " << account->GetBalance() << endl;
    cout << "Credit account balance: " << creditAccount->GetBalance() << endl;

    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;


    Circle circle(5);
    std::cout << "Obvod kruhu: " << circle.obvod() << std::endl;
    std::cout << "Obsah kruhu: " << circle.obsah() << std::endl;

    Rectangle rectangle(4, 6);
    std::cout << "Obvod obdelnika: " << rectangle.obvod() << std::endl;
    std::cout << "Obsah obdelnika: " << rectangle.obsah() << std::endl;

    Square square(4);
    std::cout << "Obvod ctverce: " << square.obvod() << std::endl;
    std::cout << "Obsah ctverce: " << square.obsah() << std::endl;

    Triangle triangle(3, 4, 5);
    std::cout << "Obvod trojuhelnika: " << triangle.obvod() << std::endl;
    std::cout << "Obsah trojuhelnika: " << triangle.obsah() << std::endl;

    return 0;
}
