#include <stdio.h>

int abs(int value){
    if(value > 0){
        return value;
    }
    if (value < 0){
        return  -value;
    }
    return value;
}

int main(){
    printf("4 abs je %d \n", abs(4));
    printf("-4 abs je %d \n", abs(-4));
    printf("0 abs je %d \n", abs(0));

    return 0;
}