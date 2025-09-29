#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

void printArguments(int argc, char* argv[]){
    for(int i = 0; i < argc; i++){
        printf("%s\n", argv[i]);
    }
}


int main(int argc, char* argv[]) {

    if (argc < 2){
        printf("malo argumentu");
        exit(1);
    }

    int n = atoi(argv[1]);
    
    for(int i = 0; i < n ; i++){
        
        printf("%d\n", rand() % 1000);
    }
    return 0;
}