#pragma once
#ifndef BANK_H
#define BANK_H

#include <string>
using namespace std;

class Client {
private:
    int code;
    string name;
    static int clientCount;

public:
    Client(int c, string n);
    ~Client();
    int GetCode() const;
    string GetName() const;
    static int GetClientCount();
};

class Account {
private:
    int number;
    double balance;
    double interestRate;
    Client* owner;
    Client* partner;
    static int accountCount;
    static double defaultInterestRate;

public:
    Account(int n, Client* c);
    Account(int n, Client* c, double ir);
    Account(int n, Client* c, Client* p);
    Account(int n, Client* c, Client* p, double ir);
    ~Account();

    int GetNumber() const;
    double GetBalance() const;
    double GetInterestRate() const;
    Client* GetOwner() const;
    Client* GetPartner() const;
    bool CanWithdraw(double a) const;
    void Deposit(double a);
    bool Withdraw(double a);
    void AddInterest();

    static int GetAccountCount();
    static double GetDefaultInterestRate();
    static void SetDefaultInterestRate(double ir);
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
    Client* CreateClient(int c, const string& n);
    Account* CreateAccount(int n, Client* c);
    Account* CreateAccount(int n, Client* c, double ir);
    Account* CreateAccount(int n, Client* c, Client* p);
    Account* CreateAccount(int n, Client* c, Client* p, double ir);
    void AddInterest();

    static int GetTotalClients();
    static int GetTotalAccounts();
};

#endif // BANK_H
