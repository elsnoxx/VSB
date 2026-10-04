#include <stdio.h>
#include <string.h>
#include <stdlib.h>

// inicializace patna na .
void inicializace_platna(int rows, int cols, int *obrazek)
{
    for (int i = 0; i < rows * cols; i++)
    {
        obrazek[i] = '.';
    }
}
// vykresli platno pruchodu zelvy
void vypis(int rows, int cols, int *obrazek)
{
    int cnt = 0;
    for (int i = 0; i < rows * cols; i++)
    {
        if (cnt == cols)
        {
            printf("\n");
            cnt = 0;
        }
        cnt += 1;

        printf("%c", obrazek[i]);
    }
    printf("\n");
}

void pozice(int cols, int x, int y, int *obrazek)
{
    // used for debugging
    // printf("tisk %d\n", x*cols+y);
    obrazek[x * cols + y] = 'o';
    printf("Zelva osa x %d, osa y %d\n",x,y);
}

int main()
{
    // vstupni promenne
    int rows = 0;
    int cols = 0;
    char znak;
    scanf("%d %d", &rows, &cols);

    int *pamet_obrazku = (int *)malloc(rows * cols * sizeof(int));

    inicializace_platna(rows, cols, pamet_obrazku);

    // osy kde se zeva nachazi
    int pocet_zelv = 0;
    int osax[3] = {0, 0, 0};
    int osay[3] = {0, 0, 0};
    char smer[] = {'r', 'r', 'r'};
    while (1)
    {
        scanf("%c", &znak);
        // used for debugging
        // printf("%c\n",znak);
        // for (int i = 0; i <= pocet_zelv; i++)
        // {
        //     printf("Zelva %d , x %d y %d smer %c\n", i, osax[i], osay[i], smer[i]);
        // }
        // printf("\n");
        vypis(rows, cols, pamet_obrazku);
        printf("%c \n",znak);

        switch (znak)
        {
        case 'x':
            vypis(rows, cols, pamet_obrazku);
            free(pamet_obrazku);
            return 0;

        case 'm':
            for (int i = 0; i <= pocet_zelv; i++)
            {
                if (smer[i] == 'l')
                {
                    osay[i] -= 1;
                    if (osay[i] < 0)
                    {
                        osay[i] = cols - 1;
                    }
                }
                else if (smer[i] == 'r')
                {
                    osay[i] += 1;
                    if (osay[i] > cols - 1)
                    {
                        osay[i] = 0;
                    }
                }
                else if (smer[i] == 'n')
                {
                    osax[i] -= 1;
                    if (osax[i] < 0)
                    {
                        osax[i] = rows - 1;
                    }
                }
                else if (smer[i] == 'd')
                {
                    osax[i] += 1;
                    if (osax[i] > rows - 1)
                    {
                        osax[i] = 0;
                    }
                }
            }

            break;
        case 'o':
            for (int i = 0; i <= pocet_zelv; i++)
            {
                // printf("Zelva %d pozice x %d y %d\n",i,osax[i],osay[i]);
                pozice(cols, osax[i], osay[i], pamet_obrazku);
            }
            break;
        case 'l':
            for (int i = 0; i <= pocet_zelv; i++)
            {
                if (smer[i] == 'r')
                {
                    smer[i] = 'n';
                }
                else if (smer[i] == 'n')
                {
                    smer[i] = 'l';
                }
                else if (smer[i] == 'l')
                {
                    smer[i] = 'd';
                }
                else if (smer[i] == 'd')
                {
                    smer[i] = 'r';
                }
            }

            break;
        case 'r':
            for (int i = 0; i <= pocet_zelv; i++)
            {
                if (smer[i] == 'r')
                {
                    smer[i] = 'd';
                }
                else if (smer[i] == 'n')
                {
                    smer[i] = 'r';
                }
                else if (smer[i] == 'l')
                {
                    smer[i] = 'n';
                }
                else if (smer[i] == 'd')
                {
                    smer[i] = 'l';
                }
            }
            break;
        case 'f':
            if (pocet_zelv < 2)
            {
                pocet_zelv += 1;
            }

            break;
        }
    }
    free(pamet_obrazku);
    return 0;
}