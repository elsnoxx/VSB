#include "car.h"
#include <iostream>

// Implementace metod třídy Auto
Auto::Auto(string zn, string mdl, int rok) : znacka(zn), model(mdl), rokVyroby(rok) {}

void Auto::Informace() const {
    cout << "Znacka: " << znacka << ", Model: " << model << ", Rok vyroby: " << rokVyroby;
}

// Implementace metod třídy OsobniAuto
OsobniAuto::OsobniAuto(string zn, string mdl, int rok, int sedadla, int rychlost) : Auto(zn, mdl, rok), pocetSedadel(sedadla), maxRychlost(rychlost) {}

void OsobniAuto::Informace() const {
    Auto::Informace();
    cout << ", Pocet sedadel: " << pocetSedadel << ", Maximalni rychlost: " << maxRychlost << " km/h" << endl;
}

// Implementace metod třídy NakladniAuto
NakladniAuto::NakladniAuto(string zn, string mdl, int rok, int nos, string typ) : Auto(zn, mdl, rok), nosnost(nos), typNakladu(typ) {}

void NakladniAuto::Informace() const {
    Auto::Informace();
    cout << ", Nosnost: " << nosnost << " kg, Typ nakladu: " << typNakladu << endl;
}
