#ifndef OSOBA_H
#define OSOBA_H

#include <string>
using namespace std;

/**
 * @class Osoba
 * @brief Třída představuje osobu se základními údaji jako jméno, příjmení a rodné číslo.
 */
class Osoba
{
private:
    string jmeno;             /**< Uchovává křestní jméno osoby */
    string prijmeni;          /**< Uchovává příjmení osoby */
    string rodne_cislo;       /**< Uchovává rodné číslo osoby */

public:
    /**
     * @brief Konstruktor pro inicializaci objektu Osoba s danými hodnotami.
     * @param jmeno Křestní jméno osoby.
     * @param prijmeni Příjmení osoby.
     * @param rodne_cislo Rodné číslo osoby.
     */
    Osoba(string jmeno, string prijmeni, string rodne_cislo);

    /**
     * @brief Vrací křestní jméno osoby.
     * @return Křestní jméno osoby.
     */
    string GetJmeno();

    /**
     * @brief Vrací příjmení osoby.
     * @return Příjmení osoby.
     */
    string GetPrijmeni();

    /**
     * @brief Vrací datum narození osoby ve formátu YYYY-MM-DD.
     * @return Datum narození osoby.
     */
    string GetDatumNarozeni();

    /**
     * @brief Pomocná funkce pro extrakci data narození z rodného čísla.
     * @param rodne_cislo Rodné číslo osoby.
     * @return Datum narození ve formátu YYYY-MM-DD.
     */
    string ExtractDatumNarozeni(string rodne_cislo);
};

#endif // OSOBA_H
