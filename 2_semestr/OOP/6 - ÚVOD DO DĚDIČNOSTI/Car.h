#pragma once
#ifndef AUTO_H
#define AUTO_H

#include <string>
using namespace std;

/**
 * @class Auto
 * @brief Bázová třída reprezentující automobil s určitou značkou, modelem a rokem výroby.
 */
class Auto {
protected:
    string znacka;     /**< Značka automobilu */
    string model;      /**< Model automobilu */
    int rokVyroby;     /**< Rok výroby automobilu */

public:
    /**
     * @brief Konstruktor pro inicializaci automobilu s danou značkou, modelem a rokem výroby.
     * @param zn Značka automobilu.
     * @param mdl Model automobilu.
     * @param rok Rok výroby automobilu.
     */
    Auto(string zn, string mdl, int rok);

    /**
     * @brief Vypíše informace o automobilu.
     */
    void Informace() const;
};

/**
 * @class OsobniAuto
 * @brief Odvozená třída reprezentující osobní automobil s určitým počtem sedadel a maximální rychlostí.
 */
class OsobniAuto : public Auto {
private:
    int pocetSedadel;    /**< Počet sedadel v osobním automobilu */
    int maxRychlost;     /**< Maximální rychlost osobního automobilu */

public:
    /**
     * @brief Konstruktor pro inicializaci osobního automobilu s danou značkou, modelem, rokem výroby, počtem sedadel a maximální rychlostí.
     * @param zn Značka automobilu.
     * @param mdl Model automobilu.
     * @param rok Rok výroby automobilu.
     * @param sedadla Počet sedadel v osobním automobilu.
     * @param rychlost Maximální rychlost osobního automobilu.
     */
    OsobniAuto(string zn, string mdl, int rok, int sedadla, int rychlost);

    /**
     * @brief Vypíše informace o osobním automobilu.
     */
    void Informace() const;
};

/**
 * @class NakladniAuto
 * @brief Odvozená třída reprezentující nákladní automobil s určitou nosností a typem nakládaného zboží.
 */
class NakladniAuto : public Auto {
private:
    int nosnost;         /**< Nosnost nákladního automobilu */
    string typNakladu;   /**< Typ nakládaného zboží */

public:
    /**
     * @brief Konstruktor pro inicializaci nákladního automobilu s danou značkou, modelem, rokem výroby, nosností a typem nakládaného zboží.
     * @param zn Značka automobilu.
     * @param mdl Model automobilu.
     * @param rok Rok výroby automobilu.
     * @param nos Nosnost nákladního automobilu.
     * @param typ Typ nakládaného zboží.
     */
    NakladniAuto(string zn, string mdl, int rok, int nos, string typ);

    /**
     * @brief Vypíše informace o nákladním automobilu.
     */
    void Informace() const;
};

#endif // AUTO_H
