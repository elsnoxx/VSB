#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

bool IsNumberValid(long number) {
    // Kontrola délky rodného čísla (9 nebo 10 číslic)
    int length = 0;
    long temp = number;
    while (temp > 0) {
        temp /= 10;
        length++;
    }
    if (length != 9 && length != 10) {
        return false;
    }

    // Kontrola dělitelnosti 11
    if (number % 11 != 0) {
        return false;
    }

    // Extrakce jednotlivých částí rodného čísla
    int day = (number / 1000000) % 100;
    int month = (number / 100000000) % 100;

    // Korekce měsíce pro ženy (přičteno 50)
    if (month > 50) {
        month -= 50;
    }

    // Kontrola platnosti data
    if (month < 1 || month > 12 || day < 1 || day > 31) {
        return false;
    }

    // Pokud všechny podmínky splněny, rodné číslo je validní
    return true;
}