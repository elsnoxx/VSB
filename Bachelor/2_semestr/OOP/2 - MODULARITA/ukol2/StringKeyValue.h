#pragma once
#ifndef STRINGKEYVALUE_H
#define STRINGKEYVALUE_H

#include <string>
using namespace std;

/**
 * @class StringKeyValue
 * @brief Třída představuje uzel binárního stromu s klíčem a hodnotou typu string.
 */
class StringKeyValue {
private:
    string key;                 /**< Klíč uzlu */
    string value;               /**< Hodnota uzlu */
    StringKeyValue* left;       /**< Ukazatel na levý poduzel */
    StringKeyValue* right;      /**< Ukazatel na pravý poduzel */

public:
    /**
     * @brief Konstruktor pro inicializaci uzlu s daným klíčem a hodnotou.
     * @param k Klíč uzlu.
     * @param v Hodnota uzlu.
     */
    StringKeyValue(string k, string v);

    /**
     * @brief Destruktor uvolňuje dynamicky alokované poduzly.
     */
    ~StringKeyValue();

    /**
     * @brief Vrací klíč uzlu.
     * @return Klíč uzlu.
     */
    string GetKey();

    /**
     * @brief Vrací hodnotu uzlu.
     * @return Hodnota uzlu.
     */
    string GetValue();

    /**
     * @brief Vrací ukazatel na levý poduzel.
     * @return Ukazatel na levý poduzel.
     */
    StringKeyValue* GetLeft();

    /**
     * @brief Vrací ukazatel na pravý poduzel.
     * @return Ukazatel na pravý poduzel.
     */
    StringKeyValue* GetRight();

    /**
     * @brief Vytvoří nový levý poduzel s daným klíčem a hodnotou.
     * @param k Klíč nového levého poduzlu.
     * @param v Hodnota nového levého poduzlu.
     * @return Ukazatel na nový levý poduzel.
     */
    StringKeyValue* CreateLeft(string k, string v);

    /**
     * @brief Vytvoří nový pravý poduzel s daným klíčem a hodnotou.
     * @param k Klíč nového pravého poduzlu.
     * @param v Hodnota nového pravého poduzlu.
     * @return Ukazatel na nový pravý poduzel.
     */
    StringKeyValue* CreateRight(string k, string v);

    /**
     * @brief Vytiskne strom uzlů do konzole s odsazením podle úrovně.
     * @param level Aktuální úroveň stromu (výchozí hodnota je 0).
     */
    void PrintTree(int level = 0);
};

#endif // STRINGKEYVALUE_H
