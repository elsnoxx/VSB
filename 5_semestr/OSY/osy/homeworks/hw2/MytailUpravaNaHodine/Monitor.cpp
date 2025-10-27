#include "Monitor.h"

void getFileInfo(const vector<string> &files, const char* logFileName) {
    int n = files.size();
    int roura[n][2];
    
    // Vytvoření rour před fork()
    for (int i = 0; i < n; i++) {
        if (pipe(roura[i]) == -1) {
            perror("pipe");
            exit(1);
        }
    }

    //logining pipe 
    int logpipe[2];
    if (pipe(logpipe) == -1) {
        perror("logpipe");
        exit(1);
    }

    
    // Vytvoření potomků pro každý soubor
    for (int i = 0; i < n; i++) {
        pid_t pid = fork();
        
        if (pid == 0) {
            // Potomek - zavře všechny nepotřebné konce rour
            for (int j = 0; j < n; j++) {
                if (j == i) {
                    close(roura[j][1]); // Zavře write konec své roury
                } else {
                    close(roura[j][0]); // Zavře read konce ostatních rour
                    close(roura[j][1]); // Zavře write konce ostatních rour
                }
            }

            // zavre log pipe
            close(logpipe[0]);
            
            // Nejprve vypíše základní informace
            printf("Potomek PID %d zpracovává soubor: %s\n", getpid(), files[i].c_str());
            getInfo(files[i].c_str());
            
            // Sleduje změny souboru
            struct stat last_stat;
            stat(files[i].c_str(), &last_stat);
            
            while(1) {
                char buffer[1414];
                int r = read(roura[i][0], buffer, sizeof(buffer) - 1);
                if (r <= 0) {
                    exit(0);
                }
                buffer[r] = '\0';
                
                if (strcmp(buffer, "check\n") == 0) {
                    struct stat current_stat;
                    if (stat(files[i].c_str(), &current_stat) == 0) {
                        if (current_stat.st_size > last_stat.st_size) {
                            printf("Soubor %s se změnil!\n", files[i].c_str());
                            getInfo(files[i].c_str());
                            char logmsg[1024];
                            snprintf(logmsg, sizeof(logmsg), "PID %d: Soubor %s se změnil\n", getpid(), files[i].c_str());
                            write(logpipe[1], logmsg, strlen(logmsg));
                            
                            last_stat = current_stat;
                        }
                    }
                }
            }
        } else if (pid < 0) {
            perror("fork");
            exit(1);
        }
    }

    pid_t logger_pid = fork();
    if (logger_pid == 0) {
        // LOGGER POTOMEK
        close(logpipe[1]); // zavře zápisový konec
        FILE *logfile = fopen(logFileName, "a+");
        if (!logfile) {
            perror("fopen logfile");
            exit(1);
        }

        char buffer[2048];
        while (1) {
            int r = read(logpipe[0], buffer, sizeof(buffer) - 1);
            if (r <= 0){
                printf("Pipe close from logger child");
                break;
            }
            buffer[r] = '\0';          
            if (access("nolog", F_OK) == 0) {
                printf("Soubor nolog nalezen, ukončuji logger...\n");
                
            }else{
                fprintf(logfile, "%s", buffer);
                fflush(logfile);
            }
        }

        fclose(logfile);
        close(logpipe[1]);
        exit(0);
    }
    
    // Rodič zavře všechny read konce rour
    for (int i = 0; i < n; i++) {
        close(roura[i][0]);
    }
    
    // Rodič pravidelně posílá check zprávy  
    printf("Rodič PID %d začíná monitorování\n", getpid());
    while(1) { // Omezeno na 10 cyklů pro testování
        sleep(1);
        for (int i = 0; i < n; i++) {
            write(roura[i][1], "check\n", 6);
        }

        if (access("stop", F_OK) == 0) {
            printf("Soubor stop nalezen, ukončuji...\n");
            break;
        }
    }
    
    // Zavře write konce a počká na potomky
    for (int i = 0; i < n; i++) {
        close(roura[i][1]);
    }

    // zavrit loging rouru
    close(logpipe[1]);
    
    for (int i = 0; i < n; i++) {
        wait(NULL);
    }
    
    printf("Rodič PID %d končí - všichni potomci dokončeni\n", getpid());
}

