#include "KeyValues.h"

KeyValues::KeyValues(int n) {
    this->keyValues = new KeyValue*[n];
    this->count = 0;
}

KeyValues::~KeyValues() {
    for (int i = 0; i < this->count; i++) {
        delete this->keyValues[i];
    }
    delete[] this->keyValues;
}

int KeyValues::Count() {
    return this->count;
}

KeyValue* KeyValues::CreateObject(int k, double v) {
    KeyValue* newObject = new KeyValue(k, v);
    this->keyValues[this->count] = newObject;
    this->count += 1;
    return newObject;
}

KeyValue* KeyValues::SearchObject(int k) {
    for (int i = 0; i < this->count; i++) {
        if (this->keyValues[i]->GetKey() == k) {
            return this->keyValues[i];
        }
    }
    return nullptr;
}

KeyValue* KeyValues::RemoveObject(int k) {
    for (int i = 0; i < this->count; i++) {
        if (this->keyValues[i]->GetKey() == k) {
            KeyValue* removedObject = this->keyValues[i];
            for (int j = i; j < this->count - 1; j++) {
                this->keyValues[j] = this->keyValues[j + 1];
            }
            this->count -= 1;
            return removedObject;
        }
    }
    return nullptr;
}
