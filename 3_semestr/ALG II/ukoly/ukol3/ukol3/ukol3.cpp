#include "HashTable.h"

int main() {
    HashTable ht;

    // Vložení klíèù a hodnot
    ht.Insert("Richard", 25);
    ht.Insert("Petra", 32);
    ht.Insert("Daniel", 20);
    ht.Insert("Charlie", 35);

    // Zobrazení obsahu
    ht.Report();


    cout << "Vyhledani zaznamu: " << endl;
    // Vyhledání hodnoty
    int value;
    if (ht.TryGetValue("Bob", value)) {
        cout << "Hodnota pro klíè 'Bob': " << value << endl;
    }
    else {
        cout << "Klíè 'Bob' nebyl nalezen." << endl;
    }


    cout << "Smazani zaznamu: " << endl;
    // Smazání klíèe
    ht.Delete("Alice");
    ht.Report();

    cout << "Aktualiza zaznamu: "<< endl;
    //Aktualizece záznamu
    ht.Insert("Bob", 40);
    ht.Report();

    // Zobrazení faktoru naplnìní
    cout << "Faktor naplnìní: " << ht.GetLoadFactor() << endl;

    return 0;
}
