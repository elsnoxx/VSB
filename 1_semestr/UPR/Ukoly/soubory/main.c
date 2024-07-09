#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

void errory(char *hlaska)
{
    printf("%s\n", hlaska);
    exit(1);
}

void zavreni_souboru(FILE *vstup, FILE *vystup, int flag_o)
{
    fclose(vstup);
    if (flag_o == 1)
    {
        fclose(vystup);
    }
}

int main(int argc, char *argv[])
{
    int flag_o = 0;
    int flag_i = 0;
    char *output = NULL;
    char *input = NULL;
    char *needle = NULL;
    FILE *vstup = NULL;
    FILE *vystup = NULL;
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-o") == 0)
        {
            flag_o++;
            if (flag_o > 1)
            {
                errory("Parameter -o provided multiple times");
            }
            i++;
            if (argc <= i)
            {
                errory("Missing output path");
            }else
            {
                output = argv[i];
            }
        }
        else if (strcmp(argv[i], "-i") == 0)
        {
            flag_i++;
            if (flag_i > 1)
            {
                errory("Parameter -i provided multiple times");
            }
        }
        else if (input == NULL)
        {
            input = argv[i];
            // printf("-%s\n", input);
        }
        else if (needle == NULL || needle[0] == '\0')
        {
            needle = argv[i];
            // printf("--%s\n", needle);
        }
        else
        {
            errory("Too many parameters provided");
        }
    }
    if (input == NULL)
    {
        errory("Input path not provided");
    }
    else if (needle == NULL)
    {
        errory("Needle not provided");
    }

    if (input != NULL)
    {
        vstup = fopen(input, "rb");
        if (vstup == NULL)
        {
            errory("Input path not provided");
        }
    }

    if (output != NULL)
    {
        vystup = fopen(output, "w");
        if (vystup == NULL)
        {
            errory("Missing output path");
        }
    }

    char radek[101];
    int cnt = 0;

    while (fgets(radek, sizeof(radek), vstup) != NULL)
    {
        // Zde zpracuj řádek, např. vytiskni ho nebo proveď hledání

        for (int i = 0; i < (int)(strlen(radek)); i++)
        {
            cnt = 0;
            if (radek[i] != '\n')
            {
                for (int j = 0; j < (int)(strlen(needle)); j++)
                {
                    // printf("Znak na pozici radku %d: %c Znak na pozici needlu %d: %c\n", i, radek[i], j, needle[j]);
                    
                    if (radek[i + j] == needle[j] && flag_i == 0)
                    {
                        cnt++;
                    }
                    else if (radek[i + j] != '\n' && needle[j] != '\n' && radek[i + j] != '\0' && needle[j] != '\0' && tolower(radek[i + j]) == tolower(needle[j]) && flag_i == 1)
                    {
                        // printf(" radek[i+j] %c, needle[j] %c\n", radek[i + j], needle[j]);
                        cnt++;
                    }else{
                        break;
                    }
                }
                // printf("\n\n");
            }
            if (cnt == (int)(strlen(needle)))
            {
                if (flag_o == 0)
                {
                    printf("%s", radek);
                }else
                {
                    // printf("%s", radek);
                    fprintf(vystup,"%s", radek);
                }
                break;
            }
        }
    }

    // printf("flag i %d, flag o %d\n", flag_i, flag_o);

    zavreni_souboru(vstup, vystup, flag_o);
    return 0;
}