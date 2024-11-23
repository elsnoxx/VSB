#pragma once

#include <string>
#include <vector>
#include <iostream>

using namespace std;

class HashTable {
public:
    HashTable();                      // Konstruktor bez parametrù
    HashTable(const int TableSize);   // Konstruktor s velikostí tabulky
    ~HashTable();                     // Destruktor

    bool ContainsKey(const string& Key) const;
    bool TryGetValue(const string& Key, int& Value) const;
    void Insert(const string& Key, const int NewValue);
    void Delete(const string& Key);
    void Clear();

    size_t GetTableSize() const;
    size_t GetNumberOfKeys() const;
    double GetLoadFactor() const;

    void Report() const;

private:
    int HashFunction(const string& Key) const;

    struct KeyValuePair {
        string Key;
        int Value;
    };

    vector<vector<KeyValuePair>> Table; // Tabulka jako vektor vektorù
    size_t TableSize;                   // Velikost tabulky (poèet slotù)
    size_t NumberOfKeys;                // Poèet klíèù v tabulce
};
