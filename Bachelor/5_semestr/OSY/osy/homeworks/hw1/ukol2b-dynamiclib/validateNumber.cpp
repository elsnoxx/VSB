#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int vahy[] = {1, 2, 4, 8, 5, 10, 9, 7, 3, 6};

bool IsNumberValid(long number) {
    long sum = 0;
    if (number < 0){
        return false;
    } 

    for (int i = 0; i < 10; i++) {
        sum += (number % 10) * vahy[i];
        number /= 10;
    }
    if (sum % 11 != 0) {
        return false;
    }
    return true;
}