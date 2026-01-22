#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

int vahy[] = {1, 2, 4, 8, 5, 10, 9, 7, 3, 6};

bool checkIsRD(long number) {
    // Kontrola délky rodného čísla (9 nebo 10 číslic)
    int length = 0;
    long temp = number;
    while (temp > 0) {
        temp /= 10;
        length++;
    }
    if (length != 9 && length != 10) {
        // printf("problem s delkou\n");
        return false;
    }

    // Kontrola dělitelnosti 11
    if (length == 10 && number % 11 != 0) {
        // printf("problem s delitelnosti\n");
        return false;
    }

    // Extrakce jednotlivých částí rodného čísla
    // printf("%ld\n",number);
    int day = (number / 1000000) % 100;
    int month = (number / 100000000) % 100;
    // printf("day %d\n", day);
    // printf("moth %d\n", month);

    // Korekce měsíce pro ženy (přičteno 50)
    if (month >= 51 && month <= 62) {
        month -= 50;
    } else if (month >= 21 && month <= 32) {
        month -= 20;
    } else if (month >= 71 && month <= 82) {
        month -= 70;
    }

    // Kontrola platnosti data
    if (month < 1 || month > 12){
        return false;
    }
    if (day < 1 || day > 31){
        return false;
    }

    // Pokud všechny podmínky splněny, rodné číslo je validní
    return true;
}



bool checkIsBank(long accountNumber) {
    long sum = 0;
    if (accountNumber < 0){
        return false;
    } 

    for (int i = 0; i < 10; i++) {
        sum += (accountNumber % 10) * vahy[i];
        accountNumber /= 10;
    }
    if (sum % 11 != 0) {
        return false;
    }
    return true;
}

bool IsNumberValid(long accountNumber) {
    long sum = 0;
    if (checkIsBank(accountNumber) && checkIsRD(accountNumber)){
        return true;
    } 
    return false;
}