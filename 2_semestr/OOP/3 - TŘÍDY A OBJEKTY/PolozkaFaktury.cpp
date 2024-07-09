#include "PolozkaFaktury.h"

PolozkaFaktury::PolozkaFaktury(std::string nazev, double cena)
    : nazev(nazev), cena(cena) {}

std::string PolozkaFaktury::getNazev() const {
    return nazev;
}

double PolozkaFaktury::getCena() const {
    return cena;
}
