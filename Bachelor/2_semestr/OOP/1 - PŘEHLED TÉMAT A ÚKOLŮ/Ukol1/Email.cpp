#include "Email.h"

Email::Email(int id, string domain, string name) {
    this->domain = domain;
    this->id = id;
    this->name = name;
}

int Email::GetId() {
    return this->id;
}

string Email::GetDomain() {
    return this->domain;
}

string Email::GetName() {
    return this->name;
}
