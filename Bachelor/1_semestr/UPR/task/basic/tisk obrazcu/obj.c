// Program to create a simple calculator
#include <stdio.h>

int main() {
    int delka = 5;

    // raked
    for (int i = 0; i <= delka; i++){
        printf("#");
    }
    printf("\n\n\n\n"); 
    //sloupec
    for (int i = 0; i <= delka; i++){
        printf("#\n");
    }
    printf("\n\n\n\n"); 
    //ctverec
    for (int i = 0; i < delka; i++){
        for(int j = 0; j < delka; j++){
            printf("#");    
        }
        printf("\n"); 
    }
    printf("\n\n\n\n");
    
    //duty ctverec
    for (int i = 0; i <= delka; i++){
        if (i == 0 || i == delka){
            for(int j = 0; j < delka; j++){
                printf("#");    
            }    
        }
        else{
            for(int a = 1; a <= delka; a++){
                if (a == 1 || a == delka){
                    printf("#");    
                } else{
                    printf(".");
                }
            }  
        }
        
        printf("\n"); 
    }
    printf("\n\n\n\n");
    //diagonala
    int cnt = 0;
    for (int i = 1; i <= delka; i++){
        cnt ++;
        
        for(int j = 1; j <= delka; j++){
            if (j == cnt){
                printf("#");    
            }else{
                printf(".");
            }
                
        }
        printf("\n"); 
    }
    printf("\n\n\n\n");
    //diagonala
    for (int i = 1; i <= delka; i++){
        printf(".");
        
        if (i == delka){
            for(int j = 0; j < delka; j++){
                printf("#");    
            } 
        }
        printf("\n"); 
    }
    
    
}
