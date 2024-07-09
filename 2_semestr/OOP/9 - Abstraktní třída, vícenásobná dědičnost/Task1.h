
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


class AbstractAccount {
public:
    AbstractAccount();
    virtual ~AbstractAccount();

    virtual bool CanWithdraw(double a) = 0;
};

class Account : public AbstractAccount {
private:
    int number;
    double balance;
    double interestRate;

    Client* owner;

public:
    Account(int n, Client* o);
    Account(int n, Client* o, double ir);
    virtual ~Account();

    int GetNumber();
    double GetBalance();
    double GetInterestRate();
    Client* GetOwner();
    virtual bool CanWithdraw(double a);

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
    virtual ~CreditAccount();

    virtual bool CanWithdraw(double a);
};
#endif // Task1_H