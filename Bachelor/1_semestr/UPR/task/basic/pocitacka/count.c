#include <stdio.h>
#include <time.h>
#include <stdlib.h>




int main(int argc, char** argv){
    int hodnota = 0;
    int suma = 0;
    while (1)
    {
        printf("Zadej cislo: ");
        scanf("%d",&hodnota);

        if (hodnota == 0)
        {
            break;
        }else{
            suma = suma + hodnota;
        }
        
    }
    printf("Suma je: %d\n",suma);
    
    return 0;
}