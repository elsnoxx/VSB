#include <iostream>
#include <string>
#include <vector>

using namespace std;

class AbstractProduct {
private:
    string name;
    int price;

public:
    AbstractProduct(string name, int price) : name(name), price(price) {}

    string getName() const { return name; }

    int getPrice() const { return price; }

    virtual string getDetails() const = 0;
};

class Computer : public AbstractProduct {
private:
    string cpu;
    string gpu;

public:
    Computer(string name, int price, string cpu, string gpu)
        : AbstractProduct(name, price), cpu(cpu), gpu(gpu) {}

    string getDetails() const override { return "CPU: " + cpu + ", GPU: " + gpu; }
};

class MobilePhone : public AbstractProduct {
private:
    string os;

public:
    MobilePhone(string name, int price, string os)
        : AbstractProduct(name, price), os(os) {}

    string getDetails() const override { return "OS: " + os; }
};

// Nový typ produktu - Tablet
class Tablet : public AbstractProduct {
private:
    string os;
    string screenSize;

public:
    Tablet(string name, int price, string os, string screenSize)
        : AbstractProduct(name, price), os(os), screenSize(screenSize) {}

    string getDetails() const override { return "OS: " + os + ", Screen size: " + screenSize; }
};

class OrderItem {
private:
    AbstractProduct* product;
    int count;

public:
    OrderItem(AbstractProduct* product, int count) : product(product), count(count) {}

    AbstractProduct* getProduct() const { return product; }

    int getCount() const { return count; }
};

class Order {
private:
    vector<OrderItem*> items;

public:
    void addItem(OrderItem* item) { items.push_back(item); }

    int getTotalPrice() const {
        int totalPrice = 0;
        for (auto item : items) {
            totalPrice += item->getProduct()->getPrice() * item->getCount();
        }
        return totalPrice;
    }

    string getSummary() const {
        string summary = "";
        for (auto item : items) {
            summary += item->getProduct()->getName() + " (" +
                item->getProduct()->getDetails() + ") - " +
                to_string(item->getCount()) + "x\n";
        }
        return summary;
    }
};

class AbstractCustomer {
private:
    string name;
    string address;

public:
    AbstractCustomer(string name, string address)
        : name(name), address(address) {}

    string getName() const { return name; }

    string getAddress() const { return address; }
};

class UnregisteredCustomer : public AbstractCustomer {
public:
    UnregisteredCustomer(string name, string address)
        : AbstractCustomer(name, address) {}
};

class VIPCustomer : public AbstractCustomer {
public:
    VIPCustomer(string name, string address)
        : AbstractCustomer(name, address) {}
};

class AbstractRegisteredCustomer : public AbstractCustomer {
private:
    string email;

public:
    AbstractRegisteredCustomer(string name, string address, string email)
        : AbstractCustomer(name, address), email(email) {}

    string getEmail() const { return email; }
};

class RegisteredCustomer : public AbstractRegisteredCustomer {
public:
    RegisteredCustomer(string name, string address, string email)
        : AbstractRegisteredCustomer(name, address, email) {}
};

class CompanyUser : public AbstractRegisteredCustomer {
private:
    string ico;

public:
    CompanyUser(string name, string address, string email, string ico)
        : AbstractRegisteredCustomer(name, address, email), ico(ico) {}

    string getIco() const { return ico; }
};

class OrderSummary {
private:
    AbstractCustomer* customer;
    Order* order;

public:
    OrderSummary(AbstractCustomer* customer, Order* order)
        : customer(customer), order(order) {}

    string getSummary() const {
        return "Customer: " + customer->getName() + ", " + customer->getAddress() +
            "\n" + order->getSummary() +
            "Total price: " + to_string(order->getTotalPrice()) + "\n";
    }
};

class AbstractStringOutput {
public:
    virtual string toString() const = 0;
};


//Úkoly na cvičení
//• Implementujte dědičné hierarchie zákazníků a produktů podle přednášky.Jak pro zákazníky, tak pro produkty navrhněte data a chování, které potomci rozšiřují.
// 
//• Navrhněte a implementujte třídy pro objednávku a položku objednávky.
// 
//• Napište program, ve kterém vytvoříte několik různých objednávek (s různými typy zákazníků a produktů).
// 
//• Vypište obsah vytvořených objednávek na obrazovku.

int main() {
    Computer* computer =
        new Computer("Computer", 1000, "Intel i7", "Nvidia RTX 2080");
    MobilePhone* mobile = new MobilePhone("Mobile", 500, "Android");

    OrderItem* computerItem = new OrderItem(computer, 2);
    OrderItem* mobileItem = new OrderItem(mobile, 1);

    Order* order = new Order();
    order->addItem(computerItem);
    order->addItem(mobileItem);

    Order* order2 = new Order();  // Nová instance pro druhou objednávku
    order2->addItem(computerItem); // Přidání položky
    order2->addItem(mobileItem);   // Přidání položky

    RegisteredCustomer* registeredCustomer =
        new RegisteredCustomer("John Doe", "Main Street 1", "john@doe.com");
    OrderSummary* orderSummary = new OrderSummary(registeredCustomer, order);

    UnregisteredCustomer* customer = new UnregisteredCustomer("Doe John", "Main Street 2");
    OrderSummary* orderSummary2 = new OrderSummary(customer, order2);

    cout << orderSummary->getSummary() << endl;
    cout << orderSummary2->getSummary() << endl;

    Computer* VIP =
        new Computer("Computer", 2000, "Intel i7", "Nvidia RTX 3080");
    MobilePhone* VIPmobile = new MobilePhone("Mobile", 500, "Android");
    Tablet* tablet = new Tablet("Tablet", 700, "Android", "10 inch");

    OrderItem* VIPcomputerItem = new OrderItem(VIP, 2);  // Nová instance pro VIP
    OrderItem* VIPmobileItem = new OrderItem(VIPmobile, 1);  // Nová instance pro VIP
    OrderItem* VIPtabletItem = new OrderItem(tablet, 1);  // Nová instance pro VIP

    Order* VIPorder = new Order();  // Nová instance pro VIP
    VIPorder->addItem(VIPcomputerItem); // Přidání položky
    VIPorder->addItem(VIPmobileItem);   // Přidání položky
    //VIPorder->addItem(VIPtabletItem);   // Přidání položky

    VIPCustomer* vipCustomer =
        new VIPCustomer("John Doe", "Main Street 1");
    OrderSummary* VIPorderSummary = new OrderSummary(vipCustomer, VIPorder);  // Nová instance pro VIP

    cout << VIPorderSummary->getSummary() << endl;

    return 0;
}
