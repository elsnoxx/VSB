#include "Task1.h"

Account::Account(int n, Client* c) {
    this->number = n;
    this->owner = 0;
    this->balance = 0;
    this->interestRate = 0;
    this->owner = c;
}

Account::Account(int n, Client* c, double ir) {
    this->number = n;
    this->owner = 0;
    this->balance = 0;
    this->interestRate = ir;
    this->owner = c;
}

PartnerAccount::PartnerAccount(int n, Client* o, Client* p) : Account(n, o) {
    this->partner = p;
}

PartnerAccount::PartnerAccount(int n, Client* o, Client* p, double ir) : Account(n, o, ir) {
    this->partner = p;
}

Client* PartnerAccount::GetPartner() { return this->partner; }

int Account::GetNumber() { return this->number; }

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

Client::Client(string n, string a, string p) {
    this->name = n;
    this->address = a;
    this->phone = p;
}

string Client::GetName() { return this->name; }

string Client::GetAddress() { return this->address; }

string Client::GetPhone() { return this->phone; }

Bank::Bank(int c, int a) {
    this->clientsCount = c;
    this->accountsCount = a;
    this->clients = new Client * [this->clientsCount];
    this->accounts = new Account * [this->accountsCount];
}

Bank::~Bank() {
    for (int i = 0; i < this->clientsCount; i++) {
        delete this->clients[i];
    }
    delete[] this->clients;

    for (int i = 0; i < this->accountsCount; i++) {
        delete this->accounts[i];
    }
    delete[] this->accounts;
}

Client* Bank::GetClient(int c) { return this->clients[c]; }

Account* Bank::GetAccount(int n) { return this->accounts[n]; }

Client* Bank::CreateClient(int c, string n) {
    Client* newClient = new Client(n, "", "");
    this->clients[c] = newClient;
    return newClient;
}

Account* Bank::CreateAccount(int n, Client* o) {
    Account* newAccount = new Account(n, o);
    this->accounts[n] = newAccount;
    return newAccount;
}

Account* Bank::CreateAccount(int n, Client* o, double ir) {
    Account* newAccount = new Account(n, o, ir);
    this->accounts[n] = newAccount;
    return newAccount;
}

PartnerAccount* Bank::CreateAccount(int n, Client* o, Client* p) {
    PartnerAccount* newAccount = new PartnerAccount(n, o, p);
    this->accounts[n] = newAccount;
    return newAccount;
}

PartnerAccount* Bank::CreateAccount(int n, Client* o, Client* p, double ir) {
    PartnerAccount* newAccount = new PartnerAccount(n, o, p, ir);
    this->accounts[n] = newAccount;
    return newAccount;
}

void Bank::AddInterest() {
    for (int i = 0; i < this->accountsCount; i++) {
        this->accounts[i]->AddInterest();
    }
}