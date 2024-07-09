#include <stdio.h>
#include <stdlib.h>

// neco meho jako strlen
int delka_retezce(char *line)
{
    int cnt = 0;
    for (size_t i = 0; line[i] != '\0'; i++)
    {
        cnt += 1;
    }
    return cnt;
}

// muj atoi
int my_atoi(char *str)
{
    int result = 0;
    for (int i = 0; str[i] != '\n'; ++i)
    {
        result = result * 10 + str[i] - '0';
    }
    return result;
}
// vypis statistiky
void print_result(int small, int big, int spaces, int big_normalize, int small_normalize, int spaces_normalize)
{
    printf("lowercase: %d -> %d\n", small, small_normalize);
    printf("uppercase: %d -> %d\n", big, big_normalize);
    printf("spaces: %d -> %d\n", spaces, spaces_normalize);
}
// vypis radku a hlidani aby na konci nebyla mezera
void vypis_radku(char *line, int *spaces_normalize)
{
    for (int i = 0; i < delka_retezce(line); i++)
    {
        if (delka_retezce(line) - 2 == i && line[i] == ' ')
        {
            *spaces_normalize -= 1;
            break;
        }
        if (line[i] != '\n')
        {
            printf("%c", line[i]);
        }
    }
    printf("\n");
}
// pocitani vyskytu malich pismen
int small_words(char *line)
{
    int male = 0;
    for (int i = 0; i < delka_retezce(line); i++)
    {
        if (line[i] >= 97 && line[i] <= 122)
        {
            male += 1;
        }
    }
    return male;
}
// pocitani vyskytu velkych pismen
int big_words(char *line)
{
    int big = 0;
    for (int i = 0; i < delka_retezce(line); i++)
    {
        if (line[i] >= 65 && line[i] <= 90)
        {
            // printf("%c\n",line[i]);
            big += 1;
        }
    }
    return big;
}
// pocitani vyskytu mezer
int spaces_cnt(char *line)
{
    int mezera = 0;
    for (int i = 0; i < delka_retezce(line); i++)
    {
        if (line[i] == ' ')
        {
            mezera += 1;
        }
    }
    return mezera;
}
// zmenseni slova odebrani duplicit z radku
void slovo(char *line)
{
    int length = delka_retezce(line);
    if (length >= 2)
    {
        for (int i = 0; i < length - 1; i++)
        {
            if (line[i] == line[i + 1])
            {
                for (int j = i; j < length - 1; j++)
                {
                    line[j] = line[j + 1];
                }
                length--;
                i--;
            }
        }
    }
    line[length] = '\0';
}
// uprava slova podle zadani
void normalizace_slov(char *line, int zacatek, int konec)
{
    int small = 0;
    int big = 0;
    for (int i = zacatek; i < konec; i++)
    {
        if (line[i] >= 97 && line[i] <= 122)
        {
            small += 1;
        }
        if (line[i] >= 65 && line[i] <= 90)
        {
            // printf("%c\n",line[i]);
            big += 1;
        }
    }
    // printf("---small %d, big %d\n",small,big);

    if (big == 0)
    {
        for (int j = zacatek; j < konec; j++)
        {
            if (line[j] >= 97 && line[j] <= 122)
            {
                line[j] = line[j] - 32;
            }
        }
    }
    if (small == 0)
    {
        for (int j = zacatek; j < konec; j++)
        {
            if (line[j] >= 65 && line[j] <= 90)
            {
                line[j] = line[j] + 32;
            }
        }
    }
    if (big > 0)
    {
        if (line[zacatek] >= 97 && line[zacatek] <= 122)
        {
            line[zacatek] = line[zacatek] - 32;
        }
        for (int j = zacatek + 1; j < konec; j++)
        {
            if (line[j] >= 65 && line[j] <= 90)
            {
                line[j] = line[j] + 32;
            }
        }
    }
}

void mezery_pred_slovem(char *line)
{
    int kursor = 0;

    for (int i = 0; i < delka_retezce(line); i++)
    {
        if (!(line[i] == ' ' && (i == 0 || line[i - 1] == ' ')))
        {
            line[kursor++] = line[i];
        }
    }
    line[kursor] = '\0';
}

void hledani_slov(char* line)
{
    int zacatek_slova = 0;
    int konec_slova = 0;
    int found = 0;
    for (int i = 0; i < delka_retezce(line); i++)
    {
        if (line[i] != ' ')
        {
            zacatek_slova = i;
            for (int j = i; line[j] != '\0'; j++)
            {
                konec_slova = j;
                if (line[j] == ' ' || line[j] == '\n' || line[j] == '\0')
                {
                    found = 1;
                    // printf("%d \n",delka_retezce(line));
                    normalizace_slov(line, zacatek_slova, konec_slova);
                    break;
                }
            }
        }
        if (found == 1)
        {
            i = konec_slova;
        }
        found = 0;
    }
}

int main()
{
    char n[50];
    fgets(n, sizeof(n), stdin);
    // Převod na celé číslo
    int rows = my_atoi(n);
    // 50 znaků + znak konce řetězce
    char line[51];
    // promenne potrebne k zpracovani radku
    int spaces = 0;
    int big = 0;
    int small = 0;
    int big_normalize = 0;
    int small_normalize = 0;
    int spaces_normalize = 0;
    // hlavni logika
    for (int i = 0; i < rows; i++)
    {
        fgets(line, sizeof(line), stdin);
        if (rows == 0)
        {
            printf("\n");
            return 0;
        }
        // statistika pred normalizaci
        small = small_words(line);
        big = big_words(line);
        spaces = spaces_cnt(line);
        // uprava mezer mezi slovy
        mezery_pred_slovem(line);
        // pouzivane pro vyskyt slova v radku
        hledani_slov(line);
        // uprava slova odstraneni stejnych znaku
        slovo(line);
        // statistika po normalizaci
        big_normalize = big_words(line);
        small_normalize = small_words(line);
        spaces_normalize = spaces_cnt(line);
        if (i != 0)
        {
            printf("\n");
        }
        // konecny vypis radku a jeho zmeny
        vypis_radku(line, &spaces_normalize);
        print_result(small, big, spaces, big_normalize, small_normalize, spaces_normalize);
    }

    return 0;
}