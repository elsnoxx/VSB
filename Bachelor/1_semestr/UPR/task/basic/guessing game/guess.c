#include <stdio.h>
#include <time.h>
#include <stdlib.h>

int random_num(){
    // inicializace nahodneho casu
    srand(time(NULL));
    //nahodne cislo v rozmezi 0 az 200
    return (rand() % 201);
}


int main(){
    int guessing_num = random_num(); 
    int gamers_num;
    printf("Zadej cislo ");
    scanf("%d", &gamers_num);

    printf("gamers number is %d and guessing number is %d \n",gamers_num,guessing_num);

    while (1)
    {
        if (gamers_num > guessing_num)
        {
            printf("Tve cislo je vetsi\n");
        }
        if (gamers_num < guessing_num)
        {
            printf("Tve cislo je mensi\n");
        }
        printf("zkus to znova ");
        scanf("%d", &gamers_num);

        if (gamers_num == guessing_num)
        {
            printf("Gratuluji k vyhre, uhodl jsi me cislo\n");
            break;
        } 
    }
    
    

    return 0;
}