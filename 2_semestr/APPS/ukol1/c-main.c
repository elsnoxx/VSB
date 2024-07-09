#include <stdio.h>

char zvire[] = "kocka";
int xnum = 0x1F2E3D4C;
int pass[ 3 ] = { 0x65707573, 0x69727072, 0x33746176 };
long g_c_array[10] = {11, -20, 31, -41, 51, -61, 71, -81, 91, -101};
int g_c_int_array[10] = {2, -3, 5, 7, 9, -11, -19, 15, 17, 19};
int array_size = 10;
char cisla[10] = { -10, -20, -30, -40, -50, -60, -70, -80, -90, -100 };
char cisla2[10] = { 10, 20, 30, 40, 50, 60, 70, 80, 90, 100 };

// external variables
extern char g_a_zvire[];
extern char g_a_bajt0;
extern char g_a_bajt1;
extern char g_a_bajt2;
extern char g_a_bajt3;
extern char text[16];
extern int vysledek;
extern int suma;
extern int suma2;
// external function
void access_zvire();
void access_bajts();
void access_pass();
void access_pass_null();
void access_array_odd();
void access_array_neg();
void access_array_min();
void access_suma();
void access_suma2();

int main()
{
    // zadani 1 - Otočte pořadí znaků v řetězci délky 5 znaků: char zvire[] = "kocka";.
    printf("Task 1\n");
    access_zvire();
    printf( "Variables zvire= %s, g_a_zvire= %s\n", zvire, g_a_zvire );


    // zadani 2 - Rozložte proměnnou int xnum = 0x1F2E3D4C na jednotlivé bajty do proměnných char bajt0, bajt1, bajt2, bajt3;.
    //            Vypisujte výsledek v hex formátu, aby byl vidět správný výsledek.
    printf("\n\n\nTask 2\n");
    access_bajts();
    printf( "Variables xnum= %x, and bajts bajt0= %x, bajt1= %x, bajt2= %x, bajt3= %x\n", xnum, g_a_bajt0, g_a_bajt1, g_a_bajt2, g_a_bajt3 );


    // zadani 3 - Programátor si naivně uložil heslo do pole int pass[ 3 ] = { 0x65707573, 0x69727072, 0x33746176 }.
    //            Přesuňte obsah pole pass do pole char text[ 16 ] = "my empty string" and vytiskněte. Nezapomeňte na ukončovací znak řetězce.
    printf("\n\n\nTask 3\n");
    access_pass();
    printf( "Variables pass= %s\n", text);


    // zadani 4 - V poli pass z předchozího příkladu vynulujte všem prvkům horní bajt. Vypište výsledek v hex formátu.
    printf("\n\n\nTask 4\n");
    access_pass_null();
    printf( "Variables pass[0]= %x, pass[1]= %x, pass[2]= %x\n", pass[0], pass[1], pass[2]);
    



    // zadani 5 - Vytvořte si pole typu long minimálně délky 10 a inicializujte si ho kladnými i zápornými čísly. 
    //            Napište funkci, která ze všech čísel v poli udělá čísla sudá: pole[ i ] &= ~1.
    printf("\n\n\nTask 5\n");
    printf("Default array: ");
    for (int i = 0; i < 10; i++)
    {
        printf("%ld, ",g_c_array[i]);
    }
    printf("\n");
    access_array_odd();
    printf("Odd array: ");
    for (int i = 0; i < 10; i++)
    {
        printf("%ld, ",g_c_array[i]);
    }
    printf("\n");



    // zadani 6 - Otočte všem číslům z předchozího příkladu znaménko. Nepoužívejte instrukci NEG.
    printf("\n\n\nTask 6\n");
    access_array_neg();
    printf("Negate array: ");
    for (int i = 0; i < 10; i++)
    {
        printf("%ld, ",g_c_array[i]);
    }
    printf("\n");


    // zadani 7 - Vytvořte si pole typu int délky minimálně 10 a inicializujte ho kladnými i zápornými čísly.
    //            Napište funkci, která najde minimální prvek v poli a vynuluje ho.
    printf("\n\n\nTask 7\n");
    printf("Default array: ");
    for (int i = 0; i < 10; i++)
    {
        printf("%d, ",g_c_int_array[i]);
    }
    printf("\n");
    access_array_min();
    printf("Null minimum number in array: ");
    for (int i = 0; i < 10; i++)
    {
        printf("%d, ",g_c_int_array[i]);
    }
    printf("\n");
    printf("Lowest number %d \n",vysledek);


    // zadani 8 - Vytvořte si pole typu char cisla[ 10 ] = { -10, -20, -30, -40, -50, -60, -70, -80, -90, -100 }.
    //           Proveďte součet prvků pole do proměnné int tak, aby nedošlo k přetečení při sčítání. Ověřte kód funkce i na kladných číslech.
    printf("\n\n\nTask 8\n");
    access_suma();
    printf("Default negative array: ");
    for (int i = 0; i < 10; i++)
    {
        printf("%d, ",cisla[i]);
    }
    printf("\n");
    printf("Suma %d \n",suma);
    
    
    access_suma2();
    
    printf("Default possitive array: ");
    for (int i = 0; i < 10; i++)
    {
        printf("%d, ",cisla2[i]);
    }
    printf("\n");
    printf("Suma %d \n",suma2);
}

