#include "StringKeyValue.h"
#include <iostream>
#include <string>
using namespace std;

/**
 * @brief Konstruktor inicializuje členské proměnné objektu StringKeyValue.
 * @param k Klíč uzlu.
 * @param v Hodnota uzlu.
 */
StringKeyValue::StringKeyValue(string k, string v) {
    this->key = k;
    this->value = v;
    this->left = nullptr;
    this->right = nullptr;
}

/**
 * @brief Vrací klíč uzlu.
 * @return Klíč uzlu.
 */
string StringKeyValue::GetKey() {
    return this->key;
}

/**
 * @brief Vrací hodnotu uzlu.
 * @return Hodnota uzlu.
 */
string StringKeyValue::GetValue() {
    return this->value;
}

/**
 * @brief Vrací ukazatel na levý poduzel.
 * @return Ukazatel na levý poduzel.
 */
StringKeyValue* StringKeyValue::GetLeft() {
    return this->left;
}

/**
 * @brief Vrací ukazatel na pravý poduzel.
 * @return Ukazatel na pravý poduzel.
 */
StringKeyValue* StringKeyValue::GetRight() {
    return this->right;
}

/**
 * @brief Vytvoří nový levý poduzel s daným klíčem a hodnotou.
 * @param k Klíč nového levého poduzlu.
 * @param v Hodnota nového levého poduzlu.
 * @return Ukazatel na nový levý poduzel.
 */
StringKeyValue* StringKeyValue::CreateLeft(string k, string v) {
    this->left = new StringKeyValue(k, v);
    return this->left;
}

/**
 * @brief Vytvoří nový pravý poduzel s daným klíčem a hodnotou.
 * @param k Klíč nového pravého poduzlu.
 * @param v Hodnota nového pravého poduzlu.
 * @return Ukazatel na nový pravý poduzel.
 */
StringKeyValue* StringKeyValue::CreateRight(string k, string v) {
    this->right = new StringKeyValue(k, v);
    return this->right;
}

/**
 * @brief Destruktor uvolňuje dynamicky alokované poduzly.
 */
StringKeyValue::~StringKeyValue() {
    if (this->left != nullptr) {
        delete this->left;
        this->left = nullptr;
    }
    if (this->right != nullptr) {
        delete this->right;
        this->right = nullptr;
    }
}

/**
 * @brief Vytiskne strom uzlů do konzole s odsazením podle úrovně.
 * @param level Aktuální úroveň stromu (výchozí hodnota je 0).
 */
void StringKeyValue::PrintTree(int level) {
    for (int i = 0; i < level; ++i) {
        cout << "  ";
    }
    cout << "Key: " << key << ", Value: " << value << endl;
    if (left) {
        left->PrintTree(level + 1);
    }
    if (right) {
        right->PrintTree(level + 1);
    }
}
