#pragma once

class KeyValue
{
private:
    int key;
    double value;

public:
    KeyValue(int k, double v);
    int GetKey();
    double GetValue();

};