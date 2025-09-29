#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main() {
    int suma = 0;
    int num;

    while(scanf("%d", &num) == 1){
        suma += num;
    }

    printf("secteno celkem %d\n", suma);

    
    return 0;
}