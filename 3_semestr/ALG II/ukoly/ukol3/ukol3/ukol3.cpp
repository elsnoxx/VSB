#include "HashTable.h"

int main() {
    HashTable ht;

    // Vložení klíèù a hodnot
    ht.Insert("Alice", 25);
    ht.Insert("Bob", 30);
    ht.Insert("Charlie", 35);

    // Zobrazení obsahu
    ht.Report();

    // Vyhledání hodnoty
    int value;
    if (ht.TryGetValue("Bob", value)) {
        cout << "Hodnota pro klíè 'Bob': " << value << endl;
    }
    else {
        cout << "Klíè 'Bob' nebyl nalezen." << endl;
    }

    // Smazání klíèe
    ht.Delete("Alice");
    ht.Report();

    // Zobrazení faktoru naplnìní
    cout << "Faktor naplnìní: " << ht.GetLoadFactor() << endl;

    return 0;
}
