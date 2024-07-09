#include <stdio.h>

int fib(int n){
    if (n <= 1)
        return n;
    else
        return fib(n - 1) + fib(n - 2);
}

int main(){
    printf("0 fibonaci je %d \n", fib(0));
    printf("1 fibonaci je %d \n", fib(1));
    printf("2 fibonaci je %d \n", fib(2));
    printf("3 fibonaci je %d \n", fib(3));
    printf("4 fibonaci je %d \n", fib(4));
    printf("5 fibonaci je %d \n", fib(5));
    printf("6 fibonaci je %d \n", fib(6));

    return 0;
}