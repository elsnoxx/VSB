#include "Task1.h"


// Client
Client::Client(int c, std::string n) : code(c), name(n) {}

int Client::GetCode() const {
    return code;
}

std::string Client::GetName() const {
    return name;
}

// Account
Account::Account(int n, Client* c) : number(n), balance(0), interestRate(0), owner(c), partner(nullptr) {}

Account::Account(int n, Client* c, double ir) : number(n), balance(0), interestRate(ir), owner(c), partner(nullptr) {}

Account::Account(int n, Client* c, Client* p) : number(n), balance(0), interestRate(0), owner(c), partner(p) {}

Account::Account(int n, Client* c, Client* p, double ir) : number(n), balance(0), interestRate(ir), owner(c), partner(p) {}

int Account::GetNumber() const {
    return number;
}

double Account::GetBalance() const {
    return balance;
}

double Account::GetInterestRate() const {
    return interestRate;
}

Client* Account::GetOwner() const {
    return owner;
}

Client* Account::GetPartner() const {
    return partner;
}

bool Account::CanWithdraw(double a) const {
    return balance >= a;
}

void Account::Deposit(double a) {
    balance += a;
}

bool Account::Withdraw(double a) {
    if (CanWithdraw(a)) {
        balance -= a;
        return true;
    }
    return false;
}

void Account::AddInterest() {
    if (interestRate > 0) {
        double interest = balance * interestRate;
        balance += interest;
    }
}

// Bank 
Bank::Bank(int c, int a) : clientsCount(0), clientsCapacity(c), accountsCount(0), accountsCapacity(a) {
    clients = new Client * [clientsCapacity];
    accounts = new Account * [accountsCapacity];
}

Bank::~Bank() {
    for (int i = 0; i < clientsCount; i++) {
        delete clients[i];
    }
    delete[] clients;

    for (int i = 0; i < accountsCount; i++) {
        delete accounts[i];
    }
    delete[] accounts;
}

Client* Bank::GetClient(int c) const {
    for (int i = 0; i < clientsCount; i++) {
        if (clients[i]->GetCode() == c) {
            return clients[i];
        }
    }
    return nullptr;
}

Account* Bank::GetAccount(int n) const {
    for (int i = 0; i < accountsCount; i++) {
        if (accounts[i]->GetNumber() == n) {
            return accounts[i];
        }
    }
    return nullptr;
}

Client* Bank::CreateClient(int c, const std::string& n) {
    if (clientsCount < clientsCapacity) {
        Client* client = new Client(c, n);
        clients[clientsCount++] = client;
        return client;
    }
    return nullptr;
}

Account* Bank::CreateAccount(int n, Client* c) {
    if (accountsCount < accountsCapacity) {
        Account* account = new Account(n, c);
        accounts[accountsCount++] = account;
        return account;
    }
    return nullptr;
}

Account* Bank::CreateAccount(int n, Client* c, double ir) {
    if (accountsCount < accountsCapacity) {
        Account* account = new Account(n, c, ir);
        accounts[accountsCount++] = account;
        return account;
    }
    return nullptr;
}

Account* Bank::CreateAccount(int n, Client* c, Client* p) {
    if (accountsCount < accountsCapacity) {
        Account* account = new Account(n, c, p);
        accounts[accountsCount++] = account;
        return account;
    }
    return nullptr;
}

Account* Bank::CreateAccount(int n, Client* c, Client* p, double ir) {
    if (accountsCount < accountsCapacity) {
        Account* account = new Account(n, c, p, ir);
        accounts[accountsCount++] = account;
        return account;
    }
    return nullptr;
}

void Bank::AddInterest() {
    for (int i = 0; i < accountsCount; i++) {
        accounts[i]->AddInterest();
    }
}