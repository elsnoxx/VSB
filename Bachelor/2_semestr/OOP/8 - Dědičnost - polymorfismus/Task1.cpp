
#include "Task1.h"

Client::Client(string n, string a, string p) {
    this->name = n;
    this->address = a;
    this->phone = p;
}

string Client::GetName() { return this->name; }

string Client::GetAddress() { return this->address; }

string Client::GetPhone() { return this->phone; }



Account::Account(int n, Client* c) : number(n), balance(0), interestRate(0.0), owner(c) {}

Account::Account(int n, Client* c, double ir) : number(n), balance(0), interestRate(ir), owner(c) {}

int Account::GetNumber() { return this->number; }

double Account::GetBalance() { return this->balance; }

double Account::GetInterestRate() { return this->interestRate; }

Client* Account::GetOwner() { return this->owner; }

bool Account::CanWithdraw(double a) { return this->balance >= a; }

string Account::GetAccountType() { return "Standard Account"; }

void Account::Deposit(double a) { this->balance += a; }

bool Account::Withdraw(double a) {
    if (CanWithdraw(a)) {
        this->balance -= a;
        return true;
    }
    return false;
}

void Account::AddInterest() {
    double interest = this->balance * this->interestRate;
    this->balance += interest;
}



CreditAccount::CreditAccount(int n, Client* o, double c) : Account(n, o), credit(c) {}

CreditAccount::CreditAccount(int n, Client* o, double ir, double c) : Account(n, o, ir), credit(c) {}

bool CreditAccount::CanWithdraw(double a) {
    return (this->GetBalance() + this->credit >= a);
}

string CreditAccount::GetAccountType() {
    return "Credit Account";
}
