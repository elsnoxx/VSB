#include <stdio.h>
#include <time.h>
#include <stdlib.h>




int main(int argc, char** argv){
    int num1,num2;
    char znamenko;
    for (int i = 1; i < argc; i++)
    {
        
        if (strcmp(argv[i], "+") == 0)
        {
            num1 = atoi(argv[i-1]);            
            num2 = atoi(argv[i+1]);
            printf("%d\n",num1+num2);
        }
        if (strcmp(argv[i], "-") == 0)
        {
            num1 = atoi(argv[i-1]);            
            num2 = atoi(argv[i+1]);
            printf("%d\n",num1-num2);
        }
        if (strcmp(argv[i], "x") == 0)
        {
            num1 = atoi(argv[i-1]);            
            num2 = atoi(argv[i+1]);
            printf("%d\n",(num1*num2));
        }
        if (strcmp(argv[i], "/") == 0)
        {
            num1 = atoi(argv[i-1]);            
            num2 = atoi(argv[i+1]);
            printf("%d\n",num1/num2);
        }
        if (strcmp(argv[i], "*") == 0)
        {
            num1 = atoi(argv[i-1]);            
            num2 = atoi(argv[i+1]);
            printf("%d\n",(num1*num2));
        }
    }

    printf("\n");
    return 0;
}