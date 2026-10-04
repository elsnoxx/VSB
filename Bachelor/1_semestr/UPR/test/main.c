#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct
{
    char *nazev;
    int mnozstvi;
} tabulka;

typedef struct
{
    char *nazev;
    int mnozstvi;
} tabulka_receptu;

int main(int argc, char *argv[])
{
    int ingredience = 0;
    int recepty_vstup = 0;
    char buffer[101];

    ingredience = atoi(argv[1]);
    recepty_vstup = atoi(argv[2]);

    if (ingredience == 0 || recepty_vstup == 0)
    {
        printf("Zadny recept nelze uvarit :(\n");
        return 0;
    }

    tabulka *zaznamy = (tabulka *)malloc(ingredience * sizeof(tabulka));

    // printf("%d\n",ingredience);
    // printf("%d\n",recepty);
    int cnt = 0;
    for (int i = 0; i < ingredience; i++)
    {

        fgets(buffer, sizeof(buffer), stdin);
        char *token = strtok(buffer, ",");
        cnt = 0;
        while (token != NULL)
        {
            switch (cnt)
            {
            case 0:
                zaznamy[i].nazev = (char *)malloc(strlen(token) + 1);
                strcpy(zaznamy[i].nazev, token);
                break;
            case 1:
                zaznamy[i].mnozstvi = atoi(token);
                break;
            default:
                break;
            }
            cnt++;
            token = strtok(NULL, ",");
        }
    }
    tabulka_receptu *recepty_tabulka = (tabulka_receptu *)malloc(recepty_vstup * sizeof(tabulka_receptu));
    int pocet = 0;
    for (int i = 0; i < recepty_vstup; i++)
    {
        fgets(buffer, sizeof(buffer), stdin);
        char *token = strtok(buffer, ",;");
        cnt = 0;
        int tmp = 0;
        pocet = atoi(token);
        token = strtok(NULL, ",;");
        while (token != NULL)
        {
            // printf("%d %s\n",tmp, token);
            if (cnt % 2 == 0)
            {
                recepty_tabulka[tmp].nazev = (char *)malloc(strlen(token) + 1);
                strcpy(recepty_tabulka[tmp].nazev, token);
                // printf("--%s\n",recepty_tabulka[tmp].nazev);
            }else{
                recepty_tabulka[tmp].mnozstvi = atoi(token);
                // printf("--%d\n",recepty_tabulka[tmp].mnozstvi);
                tmp++;
            }          
            cnt++;
            printf("%d: %s %d\n", tmp, recepty_tabulka[0].nazev, recepty_tabulka[0].mnozstvi);
            token = strtok(NULL, ",;");
        }
    }

    // for (int j = 0; j < ingredience; j++)
    // {
    //     printf("%d: %s %d\n", j, zaznamy[j].nazev, zaznamy[j].mnozstvi);
    // }
    printf("\n");

        
    

    for (int j = 0; j < ingredience; j++)
    {
        free(zaznamy[j].nazev);
    }
    free(zaznamy);

    for (int j = 0; j < pocet; j++)
    {
        free(recepty_tabulka[j].nazev);
    }
    free(recepty_tabulka);


    return 0;
}