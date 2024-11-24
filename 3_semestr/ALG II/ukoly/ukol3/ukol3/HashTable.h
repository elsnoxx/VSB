#pragma once

#include <string>
#include <vector>
#include <iostream>

using namespace std;

class HashTable {
public:
    HashTable();
    HashTable(const int TableSize);
    ~HashTable();

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

    vector<vector<KeyValuePair>> Table;
    size_t TableSize;
    size_t NumberOfKeys;
};
