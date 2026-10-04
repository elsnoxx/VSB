#include "KeyValue.h"

KeyValue::KeyValue(int k, double v) {
    this->Key = k;
    this->value = v;
    this->next = nullptr;
}

int KeyValue::GetKey() {
    return this->Key;
}

double KeyValue::GetValue() {
    return this->value;
}

KeyValue::~KeyValue() {
    if (this->next != nullptr) {
        delete this->next;
        this->next = nullptr;
    }
}

KeyValue* KeyValue::GetNext() {
    return this->next;
}

KeyValue* KeyValue::CreateNext(int k, double v) {
    this->next = new KeyValue(k, v);
    return this->next;
}
