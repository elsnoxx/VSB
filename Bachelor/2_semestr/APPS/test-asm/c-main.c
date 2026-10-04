#include <stdio.h>


// external function
int in_range( long *tp_array, int t_n, long t_from, long t_to );
int je_mocnina( int t_X, int t_M );
void mocniny( long *tp_pole, int t_N, int t_X );



int main()
{
    // ukol 3 - Ověřte, zda je zadané číslo X mocnicnou čísla M. 
    //          Výsledek bude -1 nebo příslušná mocnina.
    int vysledek = je_mocnina(18,3);
    printf("je cislo 10 mocninou cisla 100 :  %d\n", vysledek);

    // ukol 2 - Vyplňte pole mocnimami čísla X. 
    //          Při přetečení budou další výsledky 0.
    long pole[10] = {};
    mocniny(pole, 10, 10); // upravit kontrolu preteceni
    for (int i = 0; i < 10; i++)
    {
        printf("%ld-", pole[i]);
    }
    printf("\n");

    // ukol 1 -- Spočítejte, kolik čísel v poli je v zadaném rozsahu.
    // in_range( long *tp_array, int t_n, long t_from, long t_to );
    long pole2[] = {1,2,3,4,5,6,7,8,9};
    int pocet = in_range( pole2, 9, 5, 8 );
    printf("v rangu je %d\n", pocet);


    return 0;
}

