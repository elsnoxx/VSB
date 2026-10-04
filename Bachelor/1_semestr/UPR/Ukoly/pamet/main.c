#include <stdio.h>
#include <string.h>
#include <math.h>

void nacteni_hodnot(){
    
}
// funkce pro tisk spatnych cisel
void tisk_invalid(int invalid)
{
    if (invalid != 0)
    {
        printf("invalid: ");
        for (int i = 0; i < invalid; i++)
        {
            printf("#");
        }
        printf("\n");
    }
}
// vypocet pocstu cisel
int num_digits(int cislo)
{
    return (int)(log10(cislo)) + 1;
}
// funkce pro tisk mezer a zarovnani cisel
void tisk_mezer(int konec_cisel, int i)
{
    int spaces = num_digits(konec_cisel);
    int space = num_digits(i);
    if (spaces > space)
    {
        for (int j = 0; j < spaces - space; j++)
        {
            printf(" ");
        }
    }
}

int max_Num(int invalid, int *histogram)
{
    int max = histogram[0];
    for (int i = 0; i < 9; i++)
    {
        if (max < histogram[i])
        {
            max = histogram[i];
        }
    }
    if (invalid > max)
    {
        return invalid;
    }
    else
    {
        return max;
    }
}
void no_elemets(int pocatek_Cisel)
{
    printf("i");
    for (int i = pocatek_Cisel; i <= pocatek_Cisel + 8; i++)
    {
        printf("%d", i);
    }
    printf("\n");
}

int main()
{
    // innicializace promennych
    char typ = ' ';
    int pocatek_Cisel = 0;
    int konec_cisel = 0;
    int pocet_Cisel = 0;
    int histogram[9] = {0};
    int invalid = 0;
    int cislo = 0;

    // nacteni typ histogramu
    scanf("%c", &typ);
    // nacteni poctu cisel a delka nacitanych cisel
    scanf("%d %d", &pocet_Cisel, &pocatek_Cisel);
    konec_cisel = pocatek_Cisel + 8;

    // nacteni hodnot histogramu
    for (int i = 0; i < pocet_Cisel; i++)
    {
        scanf("%d", &cislo);
        if (cislo < pocatek_Cisel || cislo > konec_cisel)
        {
            invalid++;
        }
        else
        {
            histogram[cislo - pocatek_Cisel]++;
        }
    }

    switch (typ)
    {
    case 'h':
        for (int i = pocatek_Cisel; i < konec_cisel + 1; i++)
        {
            tisk_mezer(konec_cisel, i);

            printf("%d", i);
            if (histogram[i - pocatek_Cisel] > 0)
            {
                printf(" ");
                for (int j = 0; j < histogram[i - pocatek_Cisel]; j++)
                {
                    printf("#");
                }
            }
            printf("\n");
        }
        tisk_invalid(invalid);

        break;

    case 'v':
        if (pocet_Cisel == 0)
        {
            no_elemets(pocatek_Cisel);
        }
        else
        {
            int max = max_Num(invalid, histogram);

            for (int i = 0; i < 10; i++)
            {
                if (invalid == max)
                {
                    invalid--;
                    printf("#");
                }
                for (int i = 0; i < 9; i++)
                {
                    if (histogram[i] == max)
                    {
                        histogram[i]--;
                        printf("#");
                    }
                    else
                    {
                        printf(" ");
                    }
                }
                max--;
                printf("\n");
            }
            printf("i");
            for (int i = pocatek_Cisel; i <= pocatek_Cisel + 8; i++)
            {
                printf("%d", i);
            }
            printf("\n");
        }

        break;
    default:
        printf("Neplatny mod vykresleni\n");
        return 1;
    }

    return 0;
}