#include "Osoba.h"

Osoba::Osoba(std::string jmeno, std::string adresa)
    : jmeno(jmeno), adresa(adresa) {}

std::string Osoba::getJmeno() const {
    return jmeno;
}

std::string Osoba::getAdresa() const {
    return adresa;
}
