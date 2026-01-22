#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>


int main(int argn, char** argc){
    int roura[2];
    pipe(roura);

    printf("Zaciname..\n");
    if ( fork() == 0){
        close(roura[0]);
        printf("Potomek PID %d\n", getpid());
        for(int i = 0; i < 10; i++){
            char buffer[1313];
            sprintf(buffer, "%d\n", rand() % 10000);
            write(roura[1], buffer, strlen(buffer));
            usleep(50000);
        }
        close(roura[1]);
    }
    else{
        close(roura[1]);
        while(1){
            char buffer[1414];
            int r = read(roura[0], buffer, sizeof(buffer));
            if (r == 0){
                break;
            }
            write(1,buffer,r );
            
        }
        close(roura[0]);
        printf("Rodic PID %d\n", getpid());
        wait( NULL );
        printf("Rodic PID %d konci\n", getpid());
    }

    
    return 0;
}