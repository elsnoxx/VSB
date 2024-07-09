#pragma once
#ifndef KEYVALUE_H
#define KEYVALUE_H

/**
 * @class KeyValue
 * @brief Třída představuje pár klíč-hodnota s klíčem typu int a hodnotou typu double.
 */
class KeyValue {
private:
    int key;          /**< Klíč objektu */
    double value;     /**< Hodnota objektu */

public:
    /**
     * @brief Konstruktor pro inicializaci objektu s daným klíčem a hodnotou.
     * @param k Klíč objektu.
     * @param v Hodnota objektu.
     */
    KeyValue(int k, double v);

    /**
     * @brief Vrací klíč objektu.
     * @return Klíč objektu.
     */
    int GetKey();

    /**
     * @brief Vrací hodnotu objektu.
     * @return Hodnota objektu.
     */
    double GetValue();
};

#endif // KEYVALUE_H
