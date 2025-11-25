//***************************************************************************
//
// Program example for labs in subject Operating Systems
//
// Petr Olivka, Dept. of Computer Science, petr.olivka@vsb.cz, 2017
//
// Modified version for producer-consumer semaphores task.
//
//***************************************************************************

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <fcntl.h>
#include <stdarg.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/param.h>
#include <sys/time.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <errno.h>
#include <pthread.h>
#include <semaphore.h>

#define STR_QUIT    "quit"

//***************************************************************************
// Log messages

#define LOG_ERROR   0
#define LOG_INFO    1
#define LOG_DEBUG   2

int g_debug = LOG_INFO;

void log_msg( int t_log_level, const char *t_form, ... )
{
    const char *out_fmt[] = {
            "ERR: (%d-%s) %s\n",
            "INF: %s\n",
            "DEB: %s\n" };

    if ( t_log_level && t_log_level > g_debug ) return;

    char l_buf[ 1024 ];
    va_list l_arg;
    va_start( l_arg, t_form );
    vsprintf( l_buf, t_form, l_arg );
    va_end( l_arg );

    switch ( t_log_level )
    {
    case LOG_INFO:
    case LOG_DEBUG:
        fprintf( stdout, out_fmt[ t_log_level ], l_buf );
        break;
    case LOG_ERROR:
        fprintf( stderr, out_fmt[ t_log_level ], errno, strerror( errno ), l_buf );
        break;
    }
}

//***************************************************************************
// Shared producer-consumer buffer

#define N 5
#define MAX_STR 50

char buffer[N][MAX_STR];
int in = 0, out = 0;

sem_t empty;
sem_t full;
sem_t mutex;

void put_item(const char* item)
{
    sem_wait(&empty);
    sem_wait(&mutex);

    strncpy(buffer[in], item, MAX_STR-1);
    buffer[in][MAX_STR-1] = 0;
    in = (in + 1) % N;

    sem_post(&mutex);
    sem_post(&full);
}

void get_item(char* item)
{
    sem_wait(&full);
    sem_wait(&mutex);

    strncpy(item, buffer[out], MAX_STR-1);
    item[MAX_STR-1] = 0;
    out = (out + 1) % N;

    sem_post(&mutex);
    sem_post(&empty);
}

//***************************************************************************
// Producer thread

void* producer_thread(void* arg)
{
    int sock = *(int*)arg;
    delete (int*)arg;

    char buf[MAX_STR];

    while (1)
    {
        int len = read(sock, buf, sizeof(buf)-1);
        if (len <= 0) break;

        buf[len] = 0;
        buf[strcspn(buf, "\r\n")] = 0;

        put_item(buf);
        write(sock, "OK\n", 3);
    }

    close(sock);
    return NULL;
}

//***************************************************************************
// Consumer thread

void* consumer_thread(void* arg)
{
    int sock = *(int*)arg;
    delete (int*)arg;

    char item[MAX_STR];
    char sendbuf[MAX_STR + 5];   // prostor pro jméno + \n

    while (1)
    {
        // Získáme jeden item z bufferu
        get_item(item);

        // Složíme kompletní zprávu "<item>\n"
        int msg_len = snprintf(sendbuf, sizeof(sendbuf), "%s\n", item);

        // ODESÍLÁME JEDNÍM ZÁPISEM
        write(sock, sendbuf, msg_len);

        // Čekáme na OK od klienta
        char okbuf[16];
        int len = read(sock, okbuf, sizeof(okbuf));
        if (len <= 0) break;
    }

    close(sock);
    return NULL;
}


//***************************************************************************
// Help

void help( int t_narg, char **t_args )
{
    if ( t_narg <= 1 || !strcmp( t_args[ 1 ], "-h" ) )
    {
        printf(
            "\n"
            "  Producer/Consumer Task Server\n"
            "\n"
            "  Use: %s [-h -d] port_number\n"
            "\n"
            "    -d  debug mode \n"
            "    -h  this help\n"
            "\n", t_args[ 0 ] );

        exit( 0 );
    }

    if ( !strcmp( t_args[ 1 ], "-d" ) )
        g_debug = LOG_DEBUG;
}

//***************************************************************************
// Main server loop

int main( int t_narg, char **t_args )
{
    if ( t_narg <= 1 ) help( t_narg, t_args );

    int l_port = 0;

    for ( int i = 1; i < t_narg; i++ )
    {
        if (!strcmp(t_args[i], "-d")) g_debug = LOG_DEBUG;
        if (!strcmp(t_args[i], "-h")) help(t_narg, t_args);

        if (*t_args[i] != '-' && !l_port)
            l_port = atoi(t_args[i]);
    }

    if (l_port <= 0)
    {
        log_msg(LOG_INFO, "Missing or bad port!");
        help(t_narg, t_args);
    }

    // Init semaphores
    sem_init(&mutex, 0, 1);
    sem_init(&empty, 0, N);
    sem_init(&full, 0, 0);

    log_msg(LOG_INFO, "Server will listen on port: %d.", l_port);

    // Create listening socket
    int l_sock_listen = socket(AF_INET, SOCK_STREAM, 0);
    if (l_sock_listen == -1)
    {
        log_msg(LOG_ERROR, "Unable to create socket.");
        exit(1);
    }

    int opt = 1;
    setsockopt(l_sock_listen, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in srv;
    srv.sin_family = AF_INET;
    srv.sin_port = htons(l_port);
    srv.sin_addr.s_addr = INADDR_ANY;

    if (bind(l_sock_listen, (sockaddr*)&srv, sizeof(srv)) < 0)
    {
        log_msg(LOG_ERROR, "Bind failed!");
        exit(1);
    }

    if (listen(l_sock_listen, 10) < 0)
    {
        log_msg(LOG_ERROR, "Listen failed!");
        exit(1);
    }

    log_msg(LOG_INFO, "Enter 'quit' to quit server.");

    // Poll stdin + listen socket
    pollfd pfds[2];
    pfds[0].fd = STDIN_FILENO; pfds[0].events = POLLIN;
    pfds[1].fd = l_sock_listen; pfds[1].events = POLLIN;

    // ===================== MAIN SERVER LOOP ======================
    while (1)
    {
        if (poll(pfds, 2, -1) < 0)
        {
            log_msg(LOG_ERROR, "Poll failed!");
            exit(1);
        }

        // Quit on stdin
        if (pfds[0].revents & POLLIN)
        {
            char buf[128];
            int len = read(STDIN_FILENO, buf, sizeof(buf));
            if (len > 0 && !strncmp(buf, STR_QUIT, strlen(STR_QUIT)))
            {
                log_msg(LOG_INFO, "Server shutting down.");
                close(l_sock_listen);
                exit(0);
            }
        }

        // New client
        if (pfds[1].revents & POLLIN)
        {
            sockaddr_in rsa;
            socklen_t rsa_size = sizeof(rsa);
            int client = accept(l_sock_listen, (sockaddr*)&rsa, &rsa_size);
            if (client < 0)
            {
                log_msg(LOG_ERROR, "Accept failed.");
                continue;
            }

            // Send Task?
            write(client, "Task?\n", 6);

            char taskbuf[64];
            int len = read(client, taskbuf, sizeof(taskbuf)-1);
            if (len <= 0)
            {
                close(client);
                continue;
            }

            taskbuf[len] = 0;
            taskbuf[strcspn(taskbuf, "\r\n")] = 0;

            pthread_t tid;

            if (!strcasecmp(taskbuf, "producer"))
            {
                pthread_create(&tid, NULL, producer_thread, new int(client));
                pthread_detach(tid);
                log_msg(LOG_INFO, "New PRODUCER connected.");
            }
            else if (!strcasecmp(taskbuf, "consumer"))
            {
                pthread_create(&tid, NULL, consumer_thread, new int(client));
                pthread_detach(tid);
                log_msg(LOG_INFO, "New CONSUMER connected.");
            }
            else
            {
                write(client, "Unknown role\n", 13);
                close(client);
            }
        }
    }

    return 0;
}
