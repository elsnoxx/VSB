#include "Task1.h"

// Client
Client::Client(string n, string a, string p) {
    this->name = n;
    this->address = a;
    this->phone = p;
}

string Client::GetName() { return this->name; }

string Client::GetAddress() { return this->address; }

string Client::GetPhone() { return this->phone; }

// Account

Account::Account(int n, Client* c) {
    this->number = n;
    this->balance = 0;
    this->owner = c;
}

Account::Account(int n, Client* c, double ir) {
    this->number = n;
    this->balance = 0;
    this->interestRate = ir;
    this->owner = c;
}

double Account::GetBalance() { return this->balance; }

double Account::GetInterestRate() { return this->interestRate; }

Client* Account::GetOwner() { return this->owner; }

bool Account::CanWithdraw(double a) { return this->balance >= a; }

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


// CreditAccount
CreditAccount::CreditAccount(int n, Client* o, double c) : Account(n, o) {
    this->credit = c;
}

CreditAccount::CreditAccount(int n, Client* o, double ir, double c) : Account(n, o, ir) {
    this->credit = c;
}

bool CreditAccount::Withdraw(double a) {
    bool success = false;
    if (this->CanWithdraw(a)) {
        this->credit -= a;
        success = true;
    }
    return success;
}

bool CreditAccount::CanWithdraw(double a) {
    return this->credit >= a;
}