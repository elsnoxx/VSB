#pragma once
#ifndef KEYVALUE_H
#define KEYVALUE_H

class KeyValue
{
private:
    int Key;
    double value;
    KeyValue* next;

public:
    KeyValue(int k, double value);
    ~KeyValue();
    int GetKey();
    double GetValue();
    KeyValue* GetNext();
    KeyValue* CreateNext(int k, double v);
};

#endif // KEYVALUE_H
