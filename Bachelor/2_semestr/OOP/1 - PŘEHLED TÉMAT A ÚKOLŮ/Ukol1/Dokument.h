#pragma once
#ifndef DOKUMENT_H
#define DOKUMENT_H

#include <string>
#include <ctime>
using namespace std;

/**
 * @class Dokument
 * @brief Třída představuje dokument s typem, datem vytvoření a číslem.
 */
class Dokument
{
public:
    string typ;                 /**< Typ dokumentu */
    time_t datum_vytvoreni;     /**< Datum vytvoření dokumentu */
    int cislo;                  /**< Číslo dokumentu */

    /**
     * @brief Konstruktor pro inicializaci objektu Dokument s danými hodnotami.
     * @param typ Typ dokumentu.
     * @param cislo Číslo dokumentu.
     */
    Dokument(string typ, int cislo);

    /**
     * @brief Vrací typ dokumentu.
     * @return Typ dokumentu.
     */
    string GetTyp();

    /**
     * @brief Vrací datum vytvoření dokumentu ve formátu YYYY-MM-DD.
     * @return Datum vytvoření dokumentu.
     */
    string GetDatumVytvoreni();

    /**
     * @brief Vrací číslo dokumentu.
     * @return Číslo dokumentu.
     */
    int GetCislo();
};

#endif // DOKUMENT_H
