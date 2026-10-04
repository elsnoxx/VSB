#include "HashTable.h"

// Konstruktor bez parametrù
HashTable::HashTable() : TableSize(10), NumberOfKeys(0) {
    Table.resize(TableSize);
}

// Konstruktor s velikostí tabulky
HashTable::HashTable(const int TableSize) : TableSize(TableSize), NumberOfKeys(0) {
    Table.resize(TableSize);
}

// Destruktor
HashTable::~HashTable() {
    Clear();
}

// Hašovací funkce
int HashTable::HashFunction(const string& Key) const {
    int hash = 2320;
    for (char c : Key) {
        hash += c;
    }
    return hash % TableSize;
}

// Vložení klíèe a hodnoty
void HashTable::Insert(const string& Key, const int NewValue) {
    int index = HashFunction(Key);
    for (auto& pair : Table[index]) {
        if (pair.Key == Key) {
            pair.Value = NewValue;
            return;
        }
    }
    Table[index].push_back({ Key, NewValue });
    NumberOfKeys++;
}

// Hledání klíèe
bool HashTable::ContainsKey(const string& Key) const {
    int index = HashFunction(Key);
    for (const auto& pair : Table[index]) {
        if (pair.Key == Key) {
            return true;
        }
    }
    return false;
}

// Získání hodnoty podle klíèe
bool HashTable::TryGetValue(const string& Key, int& Value) const {
    int index = HashFunction(Key);
    for (const auto& pair : Table[index]) {
        if (pair.Key == Key) {
            Value = pair.Value;
            return true;
        }
    }
    return false;
}

// Smazání klíèe
void HashTable::Delete(const string& Key) {
    int index = HashFunction(Key);
    auto& slot = Table[index];
    for (auto it = slot.begin(); it != slot.end(); ++it) {
        if (it->Key == Key) {
            slot.erase(it);
            NumberOfKeys--;
            return;
        }
    }
}

// Vyèištìní tabulky
void HashTable::Clear() {
    for (auto& slot : Table) {
        slot.clear();
    }
    NumberOfKeys = 0;
}

// Velikost tabulky
size_t HashTable::GetTableSize() const {
    return TableSize;
}

// Poèet klíèù
size_t HashTable::GetNumberOfKeys() const {
    return NumberOfKeys;
}

// Faktor naplnìní
double HashTable::GetLoadFactor() const {
    return static_cast<double>(NumberOfKeys) / TableSize;
}

// Report obsahu
void HashTable::Report() const {
    for (size_t i = 0; i < TableSize; ++i) {
        cout << "Slot " << i << ": ";
        for (const auto& pair : Table[i]) {
            cout << "{" << pair.Key << ", " << pair.Value << "} ";
        }
        cout << endl;
    }
}
