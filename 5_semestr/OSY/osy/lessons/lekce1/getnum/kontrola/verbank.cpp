#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>


void printHelp() {
    fprintf(stderr, "Použití: -v pro výpis pouze validních čísel\n");
    fprintf(stderr, "       -h pro zobrazení nápovědy\n");
    fprintf(stderr, "       -b pro binární výstup\n");
    fprintf(stderr, "       -s pro zobrazení statistik\n");
}

int main(int argc, char* argv[]) {
    int OkNum = 0;
    int NgNum = 0;
    bool binary = false;
    long number;
    bool onlyOk = false;
    bool showStats = false;
    

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-v") == 0) {
            onlyOk = true;
        }
        else if (strcmp(argv[i], "-h") == 0) {
            printHelp();
            return 1;
        }
        else if (strcmp(argv[i], "-b") == 0) {
            printf("Číslo ve formátu binárním:\n");
            binary = true;
        }
        else if (strcmp(argv[i], "-s") == 0) {
            showStats = true;
        }
        else {
            printHelp();
            return 1;
        }
    }

    if (binary) {
        while (read(0, &number, sizeof(number)) == sizeof(number)) {
            if (IsAccountValid(number)) {
                printf("Číslo: %ld, je validni\n", number);
                OkNum++;
            } else {
                if (!onlyOk) {
                    printf("Číslo: %ld, neni validni\n", number);
                }
                NgNum++;
            }
        }
    } else {
        while (scanf("%ld", &number) == 1) {
            if (IsAccountValid(number)) {
                printf("Číslo: %ld, je validni\n", number);
                OkNum++;
            } else {
                if (!onlyOk) {
                    printf("Číslo: %ld, neni validni\n", number);
                }
                NgNum++;
            }
        }
    }

    if (showStats) {
        printf("Počet platných čísel: %d\n", OkNum);
        printf("Počet neplatných čísel: %d\n", NgNum);
    }
    return 0;
}
