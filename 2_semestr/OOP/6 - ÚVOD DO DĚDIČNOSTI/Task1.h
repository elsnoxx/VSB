#ifndef BANK_H
#define BANK_H

#include "Car.h"
#include <iostream>
#include <string>
#include <vector>

class Client {
private:
    string name;
    string address;
    string phone;

public:
    Client(string n, string a, string p);
    string GetName();
    string GetAddress();
    string GetPhone();
};

class Account {
private:
    int number;
    double balance;
    double interestRate;
    Client* owner;

public:
    Account(int n, Client* o);
    Account(int n, Client* o, double ir);
    int GetNumber();
    double GetBalance();
    double GetInterestRate();
    Client* GetOwner();
    bool CanWithdraw(double a);
    void Deposit(double a);
    bool Withdraw(double a);
    void AddInterest();
};

class PartnerAccount : public Account {
private:
    Client* partner;

public:
    PartnerAccount(int n, Client* o, Client* p);
    PartnerAccount(int n, Client* o, Client* p, double ir);
    Client* GetPartner();
};

class Bank {
private:
    Client** clients;
    int clientsCount;
    Account** accounts;
    int accountsCount;

public:
    Bank(int c, int a);
    ~Bank();
    Client* GetClient(int c);
    Account* GetAccount(int n);
    Client* CreateClient(int c, string n);
    Account* CreateAccount(int n, Client*);
    Account* CreateAccount(int n, Client* o, double ir);
    PartnerAccount* CreateAccount(int n, Client*, Client* p);
    PartnerAccount* CreateAccount(int n, Client*, Client* p, double i);
    void AddInterest();
};

#endif // BANK_H