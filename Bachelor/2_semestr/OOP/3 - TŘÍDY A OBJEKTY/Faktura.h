#pragma once
#ifndef FAKTURA_H
#define FAKTURA_H

#include "Osoba.h"
#include "PolozkaFaktury.h"
#include <vector>

/**
 * @class Faktura
 * @brief Třída představuje fakturu s číslem, osobou a položkami.
 */
class Faktura {
private:
    int cislo;                             /**< Číslo faktury */
    Osoba osoba;                           /**< Osoba spojená s fakturou */
    std::vector<PolozkaFaktury> polozky;   /**< Položky faktury */

public:
    /**
     * @brief Konstruktor pro inicializaci faktury s daným číslem a osobou.
     * @param cislo Číslo faktury.
     * @param osoba Osoba spojená s fakturou.
     */
    Faktura(int cislo, const Osoba& osoba);

    /**
     * @brief Přidá položku do faktury.
     * @param polozka Položka faktury, která má být přidána.
     */
    void pridatPolozku(const PolozkaFaktury& polozka);

    /**
     * @brief Vrací celkovou cenu všech položek ve faktuře.
     * @return Celková cena všech položek.
     */
    double celkovaCena() const;
};

#endif // FAKTURA_H
