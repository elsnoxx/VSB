#pragma once
#ifndef POLOZKAFAKTURY_H
#define POLOZKAFAKTURY_H

#include <string>

/**
 * @class PolozkaFaktury
 * @brief Třída představuje položku faktury s názvem a cenou.
 */
class PolozkaFaktury {
private:
    std::string nazev;   /**< Název položky faktury */
    double cena;         /**< Cena položky faktury */

public:
    /**
     * @brief Konstruktor pro inicializaci položky faktury s daným názvem a cenou.
     * @param nazev Název položky faktury.
     * @param cena Cena položky faktury.
     */
    PolozkaFaktury(std::string nazev, double cena);

    /**
     * @brief Vrací název položky faktury.
     * @return Název položky faktury.
     */
    std::string getNazev() const;

    /**
     * @brief Vrací cenu položky faktury.
     * @return Cena položky faktury.
     */
    double getCena() const;
};

#endif // POLOZKAFAKTURY_H
