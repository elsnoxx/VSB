#include <stdio.h>

int fac(int value){
    int ret = 1;
    if(value == 1 || value == 0){
        return 1;
    }
    else{
        while (value > 0)
            {
                ret = value * ret;
                value--;
            }
        return ret;
    }
    return 1;
}

int main(){
    printf("0 factorial je %d \n", fac(1));
    printf("1 factorial je %d \n", fac(1));
    printf("4 factorial je %d \n", fac(4));
    printf("5 factorial je %d \n", fac(5));

    return 0;
}