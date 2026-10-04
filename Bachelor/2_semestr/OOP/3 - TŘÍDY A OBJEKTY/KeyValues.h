#pragma once
#ifndef KEYVALUES_H
#define KEYVALUES_H

#include "KeyValue.h"

/**
 * @class KeyValues
 * @brief Třída pro správu kolekce objektů KeyValue.
 */
class KeyValues {

private:
    KeyValue** keyValues;  /**< Pole ukazatelů na objekty KeyValue */
    int count;             /**< Počet aktuálně uložených objektů */

public:
    /**
     * @brief Konstruktor pro inicializaci kolekce s danou kapacitou.
     * @param n Kapacita kolekce.
     */
    KeyValues(int n);

    /**
     * @brief Destruktor pro uvolnění dynamicky alokovaných objektů.
     */
    ~KeyValues();

    /**
     * @brief Vytvoří a přidá nový objekt KeyValue do kolekce.
     * @param k Klíč nového objektu.
     * @param v Hodnota nového objektu.
     * @return Ukazatel na nový objekt KeyValue.
     */
    KeyValue* CreateObject(int k, double v);

    /**
     * @brief Vyhledá objekt KeyValue podle klíče.
     * @param key Klíč hledaného objektu.
     * @return Ukazatel na nalezený objekt KeyValue nebo nullptr, pokud nebyl nalezen.
     */
    KeyValue* SearchObject(int key);

    /**
     * @brief Odebere objekt KeyValue z kolekce podle klíče.
     * @param k Klíč objektu, který má být odebrán.
     * @return Ukazatel na odebraný objekt KeyValue nebo nullptr, pokud nebyl nalezen.
     */
    KeyValue* RemoveObject(int k);

    /**
     * @brief Vrací počet aktuálně uložených objektů.
     * @return Počet aktuálně uložených objektů.
     */
    int Count();
};

#endif // KEYVALUES_H
