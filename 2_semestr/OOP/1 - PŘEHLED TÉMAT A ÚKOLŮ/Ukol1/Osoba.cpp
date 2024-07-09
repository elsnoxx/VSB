#include "Osoba.h"
#include <cstdlib>  // pro funkci snprintf

Osoba::Osoba(string jmeno, string prijmeni, string rodne_cislo) {
    this->jmeno = jmeno;
    this->prijmeni = prijmeni;
    this->rodne_cislo = rodne_cislo;
}

string Osoba::GetJmeno() {
    return this->jmeno;
}

string Osoba::GetPrijmeni() {
    return this->prijmeni;
}

string Osoba::GetDatumNarozeni() {
    return ExtractDatumNarozeni(this->rodne_cislo);
}

string Osoba::ExtractDatumNarozeni(string rodne_cislo) {
    string datum_cislo = rodne_cislo.substr(0, 6);

    int rok = stoi(datum_cislo.substr(0, 2));
    if (rok > 53) {
        rok += 1900;
    }
    else {
        rok += 2000;
    }

    int den = stoi(datum_cislo.substr(4, 2));
    int mesic = stoi(datum_cislo.substr(2, 2));

    char buffer[80];
    snprintf(buffer, sizeof(buffer), "%04d-%02d-%02d", rok, mesic, den);
    return string(buffer);
}
