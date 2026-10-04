#include <stdio.h>


// external function
void long2hexstr( long t_num, char *tp_str );
int pismena(char *tp_str);
void pismenaNaVelke(char *tp_str);
void pismenaNaMale(char *tp_str);
int ukol7(long num);
long faktorial( int N );
int nejvetsi_modulo( int *tp_pole, int t_N, int t_K );
long nejmensi_56bit( long *tp_array, int t_N );



int main()
{
    // zadani 1 - Najděte v poli čísel long nejmenší číslo, které má dolní bajt nulový.
    printf("Ukol 1: nejmenší číslo s nulovým dolním bajtem z pole ");
    long tp_array[] = {199395, 352996, 0000000000000, 332517, 17179869184}; 
    for (int i = 0; i < sizeof(tp_array) / sizeof(long); i++)
    {
        if (i == (sizeof(tp_array) / sizeof(int)) -1)
        {
            printf("%ld",tp_array[i]);
        }
        else{
            printf("%ld, ",tp_array[i]);
        }
    }
    printf("je cislo %ld\n",nejmensi_56bit(tp_array, 6 ));



    // zadani 2 - Převeďte číslo long na hex string.
    char hex_str[16];
    long num = 1000;
    long2hexstr(num, hex_str);
    printf("Ukol 2: převod čísla %ld na hex 0x%s\n", num,hex_str);

    


    // zadani 3 - Zjistěte, zda je v řetězci více malý či velkým písmen.
    char pismena_arr[] = "tOhlEjeMujuKoldOAPPS";
    int citac = pismena(pismena_arr);
    printf("Úkol 3: string - %s ----->  ",pismena_arr);
    // printf("%d\n",citac);
    if (citac > 0){
        printf("Velkých písmen je více");
    }else{
        printf("Malých písmen je více");
    }
    printf(" hodnota čítače je %d\n", citac);



    // zadani 4 - Spočítejte faktoriál čísla int a výsledek vraťte jako hodnotu long. Pokud dojde k přetečení při výpočtu, bude výsledek 0.
    int fac = 5;
    printf("Ukol 4: faktorial cisla %d je %ld\n",fac, faktorial( fac ));




    // zadani 5 - Které číslo v poli čísel int má nejvyšší zbytek po dělení číslem K? Vynulujte v poli všechna čísla, která mají zbytek po dělení menší, než ten nejvyšší.
    int int_arr[] = {1,16,43,40,58,10,100};
    printf("Úkol 5: největší module je %d a upravené pole je zde:  ",nejvetsi_modulo( int_arr, 7, 10 ));
    for (int i = 0; i < sizeof(int_arr) / sizeof(int); i++)
    {
        if (i == (sizeof(int_arr) / sizeof(int)) -1)
        {
            printf("%d",int_arr[i]);
        }
        else{
            printf("%d, ",int_arr[i]);
        }
    }
    printf("\n");





    // zadani 6 - Implementujte si funkci pro převod řetězce na velká či malá písmena. Podmíněný skok využijte jen pro cyklus, pro převod znaků se snažte využít jen instrukce CMOVxx.
    char naVelke[] = "tOhlEjeMujuKoldOAPPS";
    char naMale[] = "tOhlEjeMujuKoldOAPPS";
    pismenaNaVelke(naVelke);
    pismenaNaMale(naMale);
    printf("Úkol 6: převod na velka písmena %s\n",naVelke);
    printf("Úkol 6: převod na mala písmena %s\n",naMale);




    // zadani 7 - Ověřte, zda je zadané číslo long prvočíslem.
    long prvocislo = 10921861;
    int prime_check = ukol7(prvocislo);
    // printf("%d primecheck value\n", prime_check);
    if (prime_check == 1)
        printf("Úkol 7: %ld je prvocislo.\n", prvocislo);
    else
        printf("Úkol 7: %ld neni prvocislo.\n", prvocislo);

    return 0;
}

