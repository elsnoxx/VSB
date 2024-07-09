#include <iostream>
#include <string>

#include "Task1.h"

//Úkoly na cvičení
//• Implementujte příklady z přednášky a doplňte do tříd Client a Account počítání existujících objektů.
// 
//• Navrhněte a implementujte další příklady členských položek tříd.Například stejnou úrokovou sazbu pro všechny účty,
//  kterým nebyla sazba zadána v konstruktoru a kterou lze prostřednictvím metody třídy změnit.

using namespace std;


int main() {
    Bank bank(10, 20); // Vytvoření banky s kapacitou 10 klientů a 20 účtů

    // Změna výchozí úrokové sazby
    Account::SetDefaultInterestRate(0.03);

    // Vytvoření klientů
    Client* client1 = bank.CreateClient(1, "John Smith");
    Client* client2 = bank.CreateClient(2, "Alice Johnson");

    if (client1 == nullptr || client2 == nullptr) {
        cerr << "Error creating clients." << endl;
        return 1;
    }

    // Vytvoření účtů
    Account* account1 = bank.CreateAccount(1001, client1);
    Account* account2 = bank.CreateAccount(1002, client2, 0.02);

    if (account1 == nullptr || account2 == nullptr) {
        cerr << "Error creating accounts." << endl;
        return 1;
    }

    

    // Vytvoření dalšího účtu s výchozí úrokovou sazbou
    Account* account3 = bank.CreateAccount(1003, client1);

    if (account3 == nullptr) {
        cerr << "Error creating account 3." << endl;
        return 1;
    }

    // Simulace bankovních operací
    account1->Deposit(500.0);
    account2->Deposit(1000.0);
    account2->Withdraw(200.0);
    account3->Deposit(700.0);

    // Výpis informací o účtech
    cout << "Account 1 - ID: " << account1->GetNumber() << ", Balance: " << account1->GetBalance() << ", Interest Rate: " << account1->GetInterestRate() << endl;
    cout << "Account 2 - ID: " << account2->GetNumber() << ", Balance: " << account2->GetBalance() << ", Interest Rate: " << account2->GetInterestRate() << endl;
    cout << "Account 3 - ID: " << account3->GetNumber() << ", Balance: " << account3->GetBalance() << ", Interest Rate: " << account3->GetInterestRate() << endl;

    // Přidání úroků na účty
    bank.AddInterest();

    // Výpis informací o účtech po přidání úroků
    cout << "Account 1 - Number: " << account1->GetNumber() << ", Balance: " << account1->GetBalance() << endl;
    cout << "Account 2 - Number: " << account2->GetNumber() << ", Balance: " << account2->GetBalance() << endl;
    cout << "Account 3 - Number: " << account3->GetNumber() << ", Balance: " << account3->GetBalance() << endl;

    // Výpis počtu klientů a účtů
    cout << "Total clients: " << Bank::GetTotalClients() << endl;
    cout << "Total accounts: " << Bank::GetTotalAccounts() << endl;

    return 0;
}
