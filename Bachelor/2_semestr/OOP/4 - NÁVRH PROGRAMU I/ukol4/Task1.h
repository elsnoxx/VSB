#pragma once
#ifndef BANK_H
#define BANK_H

#include <string>
#include <iostream>


class Client {
private:
    int code;
    std::string name;
public:
    Client(int c, std::string n);
    int GetCode() const;
    std::string GetName() const;
};

class Account {
private:
    int number;
    double balance;
    double interestRate;
    Client* owner;
    Client* partner;
public:
    Account(int n, Client* c);
    Account(int n, Client* c, double ir);
    Account(int n, Client* c, Client* p);
    Account(int n, Client* c, Client* p, double ir);
    int GetNumber() const;
    double GetBalance() const;
    double GetInterestRate() const;
    Client* GetOwner() const;
    Client* GetPartner() const;
    bool CanWithdraw(double a) const;
    void Deposit(double a);
    bool Withdraw(double a);
    void AddInterest();
};

class Bank {
private:
    Client** clients;
    int clientsCount;
    int clientsCapacity;
    Account** accounts;
    int accountsCount;
    int accountsCapacity;

public:
    Bank(int c, int a);
    ~Bank();
    Client* GetClient(int c) const;
    Account* GetAccount(int n) const;
    Client* CreateClient(int c, const std::string& n);
    Account* CreateAccount(int n, Client* c);
    Account* CreateAccount(int n, Client* c, double ir);
    Account* CreateAccount(int n, Client* c, Client* p);
    Account* CreateAccount(int n, Client* c, Client* p, double ir);
    void AddInterest();
};

#endif // BANK_H
