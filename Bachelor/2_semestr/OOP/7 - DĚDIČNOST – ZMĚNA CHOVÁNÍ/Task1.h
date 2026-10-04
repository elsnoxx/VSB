#pragma once
#pragma once
#ifndef TASK1_H
#define TASK1_H

#include <iostream>
#include <cmath>
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
    double interestRate;

    Client* owner;

protected:
    double balance;

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

class CreditAccount : public Account {
private:
    double credit;

public:
    CreditAccount(int n, Client* o, double c);
    CreditAccount(int n, Client* o, double ir, double c);
    bool CanWithdraw(double a);
    bool Withdraw(double a);
};

#endif // TASK1_H
