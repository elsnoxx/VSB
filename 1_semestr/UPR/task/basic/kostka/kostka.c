#include <stdio.h>
#include <stdlib.h>

int random_num(){
    // inicializace nahodneho casu
    //nahodne cislo v rozmezi 1 az 6
    return (rand() % 6 + 1);
}


int main() {
    // Write C code here
    int count = 0;
    int suma = 0;
    int cislo = 0;
    for (int i = 0 ; i < 10000;i++){
        cislo = random_num();
        
        if (cislo == 0){
            printf("%d\n",cislo);    
        }
        suma += random_num();
        count++;
    }
    printf("%d\n",count);
    printf("%d\n",suma/count);

    return 0;
}