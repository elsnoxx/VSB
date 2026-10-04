#pragma once
#ifndef OSOBA_H
#define OSOBA_H

#include <string>

/**
 * @class Osoba
 * @brief Třída představuje osobu s jménem a adresou.
 */
class Osoba {
private:
    std::string jmeno;   /**< Jméno osoby */
    std::string adresa;  /**< Adresa osoby */

public:
    /**
     * @brief Konstruktor pro inicializaci osoby s daným jménem a adresou.
     * @param jmeno Jméno osoby.
     * @param adresa Adresa osoby.
     */
    Osoba(std::string jmeno, std::string adresa);

    /**
     * @brief Vrací jméno osoby.
     * @return Jméno osoby.
     */
    std::string getJmeno() const;

    /**
     * @brief Vrací adresu osoby.
     * @return Adresa osoby.
     */
    std::string getAdresa() const;
};

#endif // OSOBA_H
