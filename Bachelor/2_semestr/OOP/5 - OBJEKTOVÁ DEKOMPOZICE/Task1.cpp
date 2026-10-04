#include "Task1.h"
#include <iostream>

// Definice statických proměnných
int Client::clientCount = 0;
int Account::accountCount = 0;
double Account::defaultInterestRate = 0.01; // výchozí úroková sazba

// Implementace metod třídy Client
Client::Client(int c, string n) : code(c), name(n) {
    clientCount++;
}

Client::~Client() {
    clientCount--;
}

int Client::GetCode() const {
    return code;
}

string Client::GetName() const {
    return name;
}

int Client::GetClientCount() {
    return clientCount;
}

// Implementace metod třídy Account
Account::Account(int n, Client* c) : number(n), balance(0), interestRate(defaultInterestRate), owner(c), partner(nullptr) {
    accountCount++;
}

Account::Account(int n, Client* c, double ir) : number(n), balance(0), interestRate(ir), owner(c), partner(nullptr) {
    accountCount++;
}

Account::Account(int n, Client* c, Client* p) : number(n), balance(0), interestRate(defaultInterestRate), owner(c), partner(p) {
    accountCount++;
}

Account::Account(int n, Client* c, Client* p, double ir) : number(n), balance(0), interestRate(ir), owner(c), partner(p) {
    accountCount++;
}

Account::~Account() {
    accountCount--;
}

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

int Account::GetAccountCount() {
    return accountCount;
}

double Account::GetDefaultInterestRate() {
    return defaultInterestRate;
}

void Account::SetDefaultInterestRate(double ir) {
    defaultInterestRate = ir;
}

// Implementace metod třídy Bank
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

Client* Bank::CreateClient(int c, const string& n) {
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

int Bank::GetTotalClients() {
    return Client::GetClientCount();
}

int Bank::GetTotalAccounts() {
    return Account::GetAccountCount();
}
