#include <stdio.h>
#include <time.h>
#include <stdlib.h>




int main(int argc, char** argv){
    float sla = 99.9;
    float sto = 100.0;
    float vysledek_minuta = (1 - sla / 100.0) * 60 * 24;
    float vysledek_tyden = (1 - sla / 100.0) * 60 * 7 * 24;
    float vysledek_mesic = (1 - sla / 100.0) * 60 * 24 * 7 * 30;
    float vysledek_rok = (1 - sla / 100.0) * 60 * 24 * 7 * 30 * 365;
    printf("SLA minuty: %f", vysledek_minuta);
    printf("SLA minuty: %f", vysledek_tyden);
    printf("SLA minuty: %f", vysledek_mesic);
    printf("SLA minuty: %f", vysledek_rok);
    printf("\n");
    return 0;
}