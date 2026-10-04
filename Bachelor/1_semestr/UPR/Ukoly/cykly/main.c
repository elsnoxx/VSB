#include <stdio.h>
int main(){
    int cislo = 0;
    int obrazec = 10;
    int a = 5;
    int b = 6;
    int cnt = 0;
    scanf("%d%d%d", &obrazec, &a, &b);
    
    
    
    
    switch (obrazec){
        //ctverec done
        case 0:
            for(int i = 1; i <= b;i++){
                for(int j = 0; j <= a-1; j++ ){
                    printf("x");
                    if (j == a-1){
                        printf("\n");
                    }
                }
            }
            break;
        //duty ctverec done
        case 1:
            for(int i = 1; i <= b;i++){
                if (i == 1 || i == b){
                   for(int j = 0; j <= a-1; j++ ){
                        printf("x");    
                    }    
 
                }else{
                   for(int j = 0; j <= a-1; j++ ){
                       if (j == 0 || j == a-1){
                            printf("x"); 
                       }else{
                           printf(" ");
                       }
                    }    
                }
                printf("\n");
            }
            break;
        //duty ctverec s cisli v radku done
        case 2:
            cislo = 0;
            for(int i = 1; i <= b;i++){
                if (i == 1 || i == b){
                   for(int j = 0; j <= a-1; j++ ){
                        printf("x");    
                    }    
                }else{
                   for(int j = 0; j <= a-1; j++ ){
                       if (j == 0 || j == a-1){
                            printf("x"); 
                       }else{
                           if (cislo > 9){
                               cislo = 0;
                           }
                           printf("%d",cislo);
                           cislo++;
                       }
                    }    
                }
                printf("\n");
            }
            break;
        //diagonala    done
        case 3:
            cnt = 0;
            for (int i = 1; i <= a; i++){
                cnt ++;
                
                for(int j = 1; j <= a; j++){
                    if (j == cnt){
                        printf("x");
                        break;    
                    }else{
                        printf(" ");
                    }
                        
                }
                printf("\n"); 
            }
            break;
        //tvar T done
        case 6:
            for (int i = 1; i <= b; i++){
                if (i == 1){
                    for(int j = 1; j <= a;j++){
                        printf("x");
                    }
                }else{
                    for(int k = 1; k <= a;k++){
                       if (k == ((a/2)+1)){
                            printf("x");
                            break; 
                        }else{
                            printf(" ");
                        }
                    }
                    
                }
                printf("\n");
            }
            break;
            
        //duty ctverec s cisli v sloupci
        case 9:
            cislo = 0;
            for(int i = 1; i <= b;i++){
                if (i == 1 || i == b){
                for(int j = 0; j <= a-1; j++ ){
                        printf("x");    
                    }    
                }
                else{
                    for(int j = 0; j <= a-1; j++ ){
                        if (j == 0 || j == a-1){
                                printf("x"); 
                        }else{
                            printf("%d", cislo);
                            cislo = (cislo + b - 2) % 10;
                        }
                    
                    }    
                }
                printf("\n");
                cislo = i-1;
            }
            break;
            
            
        default:
            printf("Neznamy obrazec\n");
            break;
    }
    return 0;
}