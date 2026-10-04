#pragma once
#ifndef Task1
#define Task1_H

#include <iostream>
#include <string>

using namespace std;

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
    virtual bool CanWithdraw(double a);
    virtual string GetAccountType(); // Nová virtuální metoda

    void Deposit(double a);
    bool Withdraw(double a);
    void AddInterest();
};

class CreditAccount : public Account {
private:
    double credit;

public:
    CreditAccount(int n, Client* o, double c);
    CreditAccount(int n, Client* o, double ir, double c);

    bool CanWithdraw(double a) override;
    string GetAccountType() override; // Přepsaná virtuální metoda
};

#endif // Task1_H