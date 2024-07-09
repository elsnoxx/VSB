#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Struktura pro uchování informací o záznamech
typedef struct
{
    int Index;
    char *name;
    double start;
    double end;
    int pocet_obchodu;
} tabulka;

// uvolneni pameti
void freeStockRecords(tabulka *zaznamy, int rows)
{
    for (int i = 0; i < rows; i++)
    {
        free(zaznamy[i].name);
    }
    free(zaznamy);
}

// funkce pro tisk cisla s _ co 3 cislice
void cislo(int pocet_obchodu)
{
    // vypocet pocetu cisel
    int tmp = pocet_obchodu;
    int cnt = 0;
    while (tmp > 0)
    {
        cnt++;
        tmp /= 10;
    }

    // vytvoreni pomocneho stringu, pomoci ktereho pak tisknu
    int length = snprintf(NULL, 0, "%d", pocet_obchodu);
    char *str = malloc(length + 1);
    snprintf(str, length + 1, "%d", pocet_obchodu);

    // finalni tisk cisla
    for (size_t i = 0; i < strlen(str); i++)
    {
        if (i > 0 && (cnt - i) % 3 == 0)
        {
            printf("_");
        }
        printf("%c", str[i]);
    }

    free(str);
}
// funkce pri tisk SVG
void print_SVG(char *ticker, tabulka *zaznamy, int rows)
{
 

    int heigth = 0;


    for (int i = 0; i < rows; i++)
    {
        heigth = zaznamy[i].end - zaznamy[i].start;
        if (strcmp(ticker, zaznamy[i].name) == 0 && heigth > 0)
        {
            printf("%d  %s  %.2f    %.2f    %.2f\n", zaznamy[i].Index,zaznamy[i].name,zaznamy[i].start,zaznamy[i].end,zaznamy[i].end - zaznamy[i].start);
        }

    }
    
}

// funkce pro tisk html stranky
void ticker_print(char *ticker, tabulka *zaznamy, int rows, int max)
{
    printf("<html>\n");
    printf("<body>\n");
    printf("<div>\n");
    if (max != 0)
    {
        for (int i = 0; i < rows; i++)
        {
            if (zaznamy[i].pocet_obchodu == max)
            {
                printf("<h1>%s: highest volume</h1>\n", zaznamy[i].name);
                printf("<div>Day: %d</div>\n", zaznamy[i].Index);
                printf("<div>Start price: %.2f</div>\n", zaznamy[i].start);
                printf("<div>End price: %.2f</div>\n", zaznamy[i].end);
                printf("<div>Volume: ");
                cislo(zaznamy[i].pocet_obchodu);
                printf("</div>\n");
            }
        }
    }
    else
    {
        printf("Ticker %s was not found\n", ticker);
    }
    printf("</div>\n");
    printf("<table>\n");
    printf("<thead>\n");
    printf("<tr><th>Day</th><th>Ticker</th><th>Start</th><th>End</th><th>Diff</th><th>Volume</th></tr>\n");
    printf("</thead>\n");
    printf("<tbody>\n");
    for (int i = rows - 1; i >= 0; i--)
    {
        if (strcmp(ticker, zaznamy[i].name) == 0)
        {
            printf("<tr>\n");
            printf("	<td><b>%d</b></td>\n", zaznamy[i].Index);
            printf("	<td><b>%s</b></td>\n", zaznamy[i].name);
            printf("	<td><b>%.2f</b></td>\n", zaznamy[i].start);
            printf("	<td><b>%.2f</b></td>\n", zaznamy[i].end);
            printf("	<td><b>%.2f</b></td>\n", zaznamy[i].end - zaznamy[i].start);
            printf("	<td><b>");
            cislo(zaznamy[i].pocet_obchodu);
            printf("</b></td>\n");
            printf("</tr>\n");
        }
        else
        {
            printf("<tr>\n");
            printf("	<td>%d</td>\n", zaznamy[i].Index);
            printf("	<td>%s</td>\n", zaznamy[i].name);
            printf("	<td>%.2f</td>\n", zaznamy[i].start);
            printf("	<td>%.2f</td>\n", zaznamy[i].end);
            printf("	<td>%.2f</td>\n", zaznamy[i].end - zaznamy[i].start);
            printf("	<td>");
            cislo(zaznamy[i].pocet_obchodu);
            printf("</td>\n");
            printf("</tr>\n");
        }
    }
    printf("</tbody>\n");
    printf("</table>\n");
    print_SVG(ticker, zaznamy, rows);
    printf("</body>\n");
    printf("</html>\n");
}

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        printf("Wrong parameters\n");
        return 1;
    }
    // Načtení názvu akcie a počtu řádků
    char *ticker = argv[1];
    int rows = atoi(argv[2]);
    int cnt = 0;

    // Alokace paměti pro záznamy
    tabulka *zaznamy = (tabulka *)malloc(rows * sizeof(tabulka));
    // nacitani dat do struktury
    char buffer[101] = {};
    for (int i = 0; i < rows; i++)
    {
        fgets(buffer, sizeof(buffer), stdin);
        char *token = strtok(buffer, ",");
        cnt = 0;
        while (token != NULL)
        {
            // printf("%s\n", token);
            switch (cnt)
            {
            case 0:
                zaznamy[i].Index = atoi(token);
                break;
            case 1:
                zaznamy[i].name = (char *)malloc(strlen(token) + 1);
                strcpy(zaznamy[i].name, token);
                break;
            case 2:
                zaznamy[i].start = atof(token);
                break;
            case 3:
                zaznamy[i].end = atof(token);
                break;
            case 4:
                zaznamy[i].pocet_obchodu = atoi(token);
                break;
            }

            cnt++;
            token = strtok(NULL, ",");
        }
    }
    // vyhledavani nejvyssi pocet obchodu
    int max = 0;
    for (int i = 0; i < rows; i++)
    {
        if (strcmp(zaznamy[i].name, ticker) == 0 && max < zaznamy[i].pocet_obchodu)
        {
            max = zaznamy[i].pocet_obchodu;
        }

        // printf("index %d, tiker %s, start %.2f, end %.2f, trades %d\n",zaznamy[i].Index,zaznamy[i].name,zaznamy[i].start,zaznamy[i].end,zaznamy[i].pocet_obchodu);
    }

    // finalni tisl
    ticker_print(ticker, zaznamy, rows, max);
    // uvolneni pameti
    freeStockRecords(zaznamy, rows);
}