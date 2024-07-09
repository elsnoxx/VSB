#include <iostream>
#include <string>

#include "Task1.h"
#include "Task2.h"

//Úkoly na cvičení
//• Implementujte příklad z přednášky a navrhněte kód, který bude používat všechny třídy.
//  Vytvořte desítky klientů a účtů v bance a nasimulujte některé běžné úkony prováděné v bance.
// 
//• Navrhněte a implementujte podobnou úlohu, jako například lékařskou ordinaci, malou školu apod.

using namespace std;


int main() {
    Bank bank(15, 20); // Vytvoření banky s kapacitou 15 klientů a 20 účtů

    // Vytvoření klientů
    Client* clients[15]; // Pole pro uchování ukazatelů na klienty

    for (int i = 0; i < 15; ++i) {
        clients[i] = bank.CreateClient(i + 1, "Client " + to_string(i + 1)); // Vytvoříme a přidáme klienty do pole
        if (clients[i] == nullptr) {
            cerr << "Error creating client " << i + 1 << endl;
            return 1;
        }
    }

    // Vytvoření účtů
    Account* accounts[15]; // Pole pro uchování ukazatelů na účty

    for (int i = 0; i < 15; ++i) {
        accounts[i] = bank.CreateAccount(1000 + i, clients[i], 100.0 * i); // Vytvoříme účty pro každého klienta
        if (accounts[i] == nullptr) {
            cerr << "Error creating account " << i + 1 << endl;
            return 1;
        }
    }

    // Simulace bankovních operací na účtech
    for (int i = 0; i < 15; ++i) {
        accounts[i]->Deposit(50.0 * (i + 1)); // Vkládáme peníze na účty
        accounts[i]->Withdraw(20.0 * (i + 1)); // Vybereme peníze z účtů
    }

    // Výpis informací o účtech
    for (int i = 0; i < 15; ++i) {
        cout << "Account " << i + 1 << " - ID: " << accounts[i]->GetNumber() << ", Balance: " << accounts[i]->GetBalance() << endl;
    }

    // Přidání úroků na účty
    bank.AddInterest();

    // Výpis informací o účtech po přidání úroků
    for (int i = 0; i < 15; ++i) {
        cout << "Account " << i + 1 << " - ID: " << accounts[i]->GetNumber() << ", Balance: " << accounts[i]->GetBalance() << endl;
    }






    Clinic clinic;

    // Vytvoření pacientů
    Patient* patient1 = clinic.AddPatient("Richard Ficek", 20);
    Patient* patient2 = clinic.AddPatient("John Doe", 25);

    // Vytvoření doktorů
    Doctor* doctor1 = clinic.AddDoctor("MUDr. Novakova", "Vseobecny lekar");
    Doctor* doctor2 = clinic.AddDoctor("MUDr. David", "Zubni lekar");

    // Naplánování schůzek
    Appointment* appointment1 = clinic.ScheduleAppointment(patient1, doctor1, "15.5.2023");
    Appointment* appointment2 = clinic.ScheduleAppointment(patient2, doctor2, "16.5.2023");

    // Výpis naplánovaných schůzek
    clinic.PrintAppointments();

    return 0;
}
