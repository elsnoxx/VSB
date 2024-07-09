#include <iostream>
#include <string>

#include "Car.h"
#include "Task1.h"

using namespace std;





int main() {

    Account* a;
    PartnerAccount* pa;
    Bank* b = new Bank(100, 1000);
    Client* o = b->CreateClient(0, "Smith");
    Client* p = b->CreateClient(1, "Jones");
    a = b->CreateAccount(0, o);
    pa = b->CreateAccount(1, 0, p);
    cout << a->GetOwner()->GetName() << endl;
    cout << pa->GetPartner()->GetName() << endl;
    cout << b->GetClient(1)->GetName() << endl;

    getchar();


    // Testování tříd

    // Vytvoření auta
    Auto auto1("Skoda", "Octavia", 2020);
    auto1.Informace();
    cout << endl;
    // Vytvoření osobního auta
    OsobniAuto osobni("Skoda", "Octavia", 2020, 5, 200);
    cout << "Informace o osobnim aute: ";
    osobni.Informace();

    // Vytvoření nákladního auta
    NakladniAuto nakladni("Volvo", "FH16", 2019, 20000, "Potraviny");
    cout << "Informace o nakladnim aute: ";
    nakladni.Informace();

    return 0;
}
