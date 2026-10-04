#include "Faktura.h"

Faktura::Faktura(int cislo, const Osoba& osoba)
    : cislo(cislo), osoba(osoba) {}

void Faktura::pridatPolozku(const PolozkaFaktury& polozka) {
    polozky.push_back(polozka);
}

double Faktura::celkovaCena() const {
    double celkem = 0.0;
    for (const PolozkaFaktury& polozka : polozky) {
        celkem += polozka.getCena();
    }
    return celkem;
}
