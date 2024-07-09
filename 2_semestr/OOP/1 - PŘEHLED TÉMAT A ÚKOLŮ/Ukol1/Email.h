#ifndef EMAIL_H
#define EMAIL_H

#include <string>
using namespace std;

/**
 * @class Email
 * @brief Třída představuje e-mail s ID, doménou a jménem.
 */
class Email
{
private:
    int id;             /**< ID e-mailu */
    string domain;      /**< Doména e-mailu */
    string name;        /**< Jméno e-mailu */

public:
    /**
     * @brief Konstruktor pro inicializaci objektu Email s danými hodnotami.
     * @param id ID e-mailu.
     * @param domain Doména e-mailu.
     * @param name Jméno e-mailu.
     */
    Email(int id, string domain, string name);

    /**
     * @brief Vrací ID e-mailu.
     * @return ID e-mailu.
     */
    int GetId();

    /**
     * @brief Vrací doménu e-mailu.
     * @return Doména e-mailu.
     */
    string GetDomain();

    /**
     * @brief Vrací jméno e-mailu.
     * @return Jméno e-mailu.
     */
    string GetName();
};

#endif // EMAIL_H
