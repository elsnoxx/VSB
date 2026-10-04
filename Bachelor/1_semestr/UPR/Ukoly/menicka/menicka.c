#include <stdio.h>

int main()
{
    int hodnota = 0;
    scanf("%d",&hodnota);
    
    int bankovka5k = hodnota / 5000;
    hodnota -= bankovka5k * 5000;
    
    int bankovka2k = hodnota / 2000;
    hodnota -= bankovka2k * 2000;
    
    int bankovka1k = hodnota / 1000;
    hodnota -= bankovka1k * 1000;
    
    int bankovka5 = hodnota / 500;
    hodnota -= bankovka5 * 500;
    
    int bankovka2 = hodnota / 200;
    hodnota -= bankovka2 * 200;
    
    int bankovka1 = hodnota / 100;
    hodnota -= bankovka1 * 100;
    
    printf("Bankovka 5000: %dx\n",bankovka5k);
    printf("Bankovka 2000: %dx\n",bankovka2k);
    printf("Bankovka 1000: %dx\n",bankovka1k);
    printf("Bankovka 500: %dx\n",bankovka5);
    printf("Bankovka 200: %dx\n",bankovka2);
    printf("Bankovka 100: %dx\n",bankovka1);
    

    return 0;
}
