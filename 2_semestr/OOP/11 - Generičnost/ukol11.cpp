#include <iostream>
#include <string>
#include <vector>
#include <list> // Přidáno pro Queue
#include <stdexcept> // Přidáno pro runtime_error

using namespace std;

// Generická třída pro zapouzdření instance objektu
template <class T>
class BOX {
private:
    T* instance;

public:
    BOX(T* i);
    T* GetInstance();
};

template <class T>
BOX<T>::BOX(T* i) {
    this->instance = i;
}

template <class T>
T* BOX<T>::GetInstance() {
    return this->instance;
}

// Třída reprezentující objekt A s hodnotou
class A {
private:
    int value;

public:
    A(int v);
    int GetValue();
};

A::A(int v) {
    this->value = v;
}

int A::GetValue() {
    return this->value;
}

// Třída reprezentující objekt B, který je odvozen od A
class B : public A {
public:
    B(int v);
};

B::B(int v) : A(v) {}

// Generická třída reprezentující položku v elektronickém obchodě
template <class Item>
class ECommerceItem {
private:
    Item product;
    double price;

public:
    ECommerceItem(const Item& product, double price) : product(product), price(price) {}

    Item getProduct() const {
        return product;
    }

    double getPrice() const {
        return price;
    }
};

// Generická třída reprezentující bankovní účet
template <class Currency>
class BankAccount {
private:
    Currency balance;

public:
    BankAccount() : balance(0) {}

    void deposit(const Currency& amount) {
        balance += amount;
    }

    void withdraw(const Currency& amount) {
        if (balance < amount) {
            throw runtime_error("Insufficient funds");
        }
        balance -= amount;
    }

    Currency getBalance() const {
        return balance;
    }
};

// Generická třída reprezentující frontu
template <class T>
class Queue {
private:
    list<T> elements;

public:
    void enqueue(const T& element) {
        elements.push_back(element);
    }

    T dequeue() {
        if (elements.empty()) {
            throw out_of_range("Queue is empty");
        }
        T element = elements.front();
        elements.pop_front();
        return element;
    }

    bool isEmpty() const {
        return elements.empty();
    }
};

// Generická třída reprezentující zásobník
template <class T>
class Stack {
private:
    vector<T> elements;

public:
    void push(const T& element) {
        elements.push_back(element);
    }

    T pop() {
        if (elements.empty()) {
            throw out_of_range("Stack is empty");
        }
        T element = elements.back();
        elements.pop_back();
        return element;
    }

    bool isEmpty() const {
        return elements.empty();
    }
};


//Úkoly na cvičení
//• Implementujte příklad z prezentace a vyzkoušejte i kombinaci generičnosti s polymorfismem.
// 
//• Implementujte datové struktury jako je zásobník, fronta apod.jako generické typy.
// 
//• Napište příklad využívající vámi implementované generické typy.
// 
//• Popřemýšlejte, jak a kde využít generičnost v příkladech s bankou a s elektronickým obchodem a popř.doplňte implementaci o práci s generickými třídami.

int main() {
    A* a = new A(50);
    B* b = new B(100);
    BOX<A>* ta = new BOX<A>(a);
    BOX<B>* tb = new BOX<B>(b);
    cout << ta->GetInstance()->GetValue() << endl;
    cout << tb->GetInstance()->GetValue() << endl;

    delete ta;
    delete tb;
    delete a;
    delete b;

    // Použití generického zásobníku pro objekty třídy A
    Stack<A> aStack;
    aStack.push(A(10));
    aStack.push(A(20));
    cout << "Popped value from A stack: " << aStack.pop().GetValue() << endl;

    // Použití generické fronty pro objekty třídy B
    Queue<B> bQueue;
    bQueue.enqueue(B(30));
    bQueue.enqueue(B(40));
    cout << "Dequeued value from B queue: " << bQueue.dequeue().GetValue() << endl;

    // Použití generické třídy BankAccount pro práci s financemi v různých měnách
    BankAccount<double> account1; // Účet v EUR
    BankAccount<double> account2;    // Účet v CZK
    account1.deposit(100.50);
    account2.deposit(5000.89);
    cout << "Account 1 balance (EUR): " << account1.getBalance() << endl;
    cout << "Account 2 balance (CZK): " << account2.getBalance() << endl;

    // Použití generické třídy ECommerceItem pro položky v elektronickém obchodě
    ECommerceItem<string> item1("Smartphone", 599.99);
    ECommerceItem<int> item2(12345, 29.99);
    cout << "Item 1: " << item1.getProduct() << ", Price: $" << item1.getPrice() << endl;
    cout << "Item 2: " << item2.getProduct() << ", Price: $" << item2.getPrice() << endl;

    return 0;
}
