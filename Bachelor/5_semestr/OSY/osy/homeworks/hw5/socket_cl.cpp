//***************************************************************************
//
// Producer / Consumer CLIENT
//
// Compatible with modified server for Operating Systems project.
//
//***************************************************************************

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <fcntl.h>
#include <stdarg.h>
#include <poll.h>
#include <pthread.h>
#include <sys/socket.h>
#include <sys/param.h>
#include <sys/types.h>
#include <sys/time.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <errno.h>
#include <netdb.h>

#define LOG_ERROR   0
#define LOG_INFO    1
#define LOG_DEBUG   2

int g_debug = LOG_INFO;
int g_speed = 60;       // jmen za minutu
int g_stop = 0;

//***************************************************************************
// Logging

void log_msg( int t_log_level, const char *t_form, ... )
{
    const char *out_fmt[] = {
        "ERR: (%d-%s) %s\n",
        "INF: %s\n",
        "DEB: %s\n"
    };

    if (t_log_level && t_log_level > g_debug) return;

    char l_buf[1024];
    va_list l_arg;
    va_start(l_arg, t_form);
    vsprintf(l_buf, t_form, l_arg);
    va_end(l_arg);

    if (t_log_level == LOG_ERROR)
        fprintf(stderr, out_fmt[t_log_level], errno, strerror(errno), l_buf);
    else
        fprintf(stdout, out_fmt[t_log_level], l_buf);
}

//***************************************************************************
// PRODUCER thread

void* producer_thread(void* arg)
{
    int sock = *(int*)arg;
    delete (int*)arg;

    FILE* f = fopen("jmena.txt", "r");
    if (!f) {
        log_msg(LOG_ERROR, "Unable to open jmena.txt");
        close(sock);
        return NULL;
    }

    char line[256];
    char sendbuf[300];

    while (!g_stop && fgets(line, sizeof(line), f))
    {
        line[strcspn(line, "\r\n")] = 0;  // remove newline
        printf("Sent: %s\n", line);
        // poslat JEDNÍM zápisem
        snprintf(sendbuf, sizeof(sendbuf), "%s\n", line);
        write(sock, sendbuf, strlen(sendbuf));

        // čekej na OK
        char ok[16];
        int len = read(sock, ok, sizeof(ok));
        if (len <= 0) break;

        // rychlost
        int delay_ms = 60000 / g_speed;
        usleep(delay_ms * 1000);
    }

    fclose(f);
    close(sock);
    return NULL;
}


//***************************************************************************
// CONSUMER thread

void* consumer_thread(void* arg)
{
    int sock = *(int*)arg;
    delete (int*)arg;

    char buf[256];

    while (1)
    {
        int len = read(sock, buf, sizeof(buf)-1);
        if (len <= 0) break;

        buf[len] = 0;

        // odstraním newline aby se výpis nelepí
        buf[strcspn(buf, "\r\n")] = 0;

        printf("RECEIVED: %s\n", buf);

        write(sock, "OK\n", 3);
    }

    close(sock);
    return NULL;
}

//***************************************************************************
// MAIN

int main( int t_narg, char **t_args )
{
    if (t_narg <= 2)
    {
        printf("Usage: %s IP PORT\n", t_args[0]);
        exit(1);
    }

    char* host = t_args[1];
    int port = atoi(t_args[2]);

    // Resolve host
    addrinfo ai_req{}, *ai_ans;
    ai_req.ai_family = AF_INET;
    ai_req.ai_socktype = SOCK_STREAM;

    if (getaddrinfo(host, NULL, &ai_req, &ai_ans))
    {
        log_msg(LOG_ERROR, "Unknown host.");
        exit(1);
    }

    sockaddr_in srv = *(sockaddr_in*)ai_ans->ai_addr;
    srv.sin_port = htons(port);
    freeaddrinfo(ai_ans);

    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0)
    {
        log_msg(LOG_ERROR, "Cannot create socket.");
        exit(1);
    }

    if (connect(sock, (sockaddr*)&srv, sizeof(srv)) < 0)
    {
        log_msg(LOG_ERROR, "Cannot connect server.");
        exit(1);
    }

    log_msg(LOG_INFO, "Connected.");

    // -------------------------------------------------------------------
    // ČEKÁNÍ NA "Task?"
    char buf[128];
    int len = read(sock, buf, sizeof(buf));
    if (len <= 0)
    {
        log_msg(LOG_ERROR, "Server closed.");
        exit(1);
    }

    buf[len] = 0;
    if (strncmp(buf, "Task?", 5) != 0)
    {
        log_msg(LOG_ERROR, "Server protocol error, expected 'Task?'");
        exit(1);
    }

    printf("Server: Task? (producer/consumer): ");
    fflush(stdout);

    char task[32];
    fgets(task, sizeof(task), stdin);
    task[strcspn(task, "\r\n")] = 0;

    write(sock, task, strlen(task));
    write(sock, "\n", 1);

    pthread_t tid;

    // ================================================================
    // PRODUCER
    if (!strcasecmp(task, "producer"))
    {
        log_msg(LOG_INFO, "You are PRODUCER.");

        pthread_create(&tid, NULL, producer_thread, new int(sock));
        pthread_detach(tid);

        // main thread: control speed
        while (1)
        {
            printf("Zadej rychlost (jmen/min): ");
            char s[32];
            if (!fgets(s, sizeof(s), stdin)) break;

            int sp = atoi(s);
            if (sp > 0) g_speed = sp;
        }
    }

    // ================================================================
    // CONSUMER
    else if (!strcasecmp(task, "consumer"))
    {
        log_msg(LOG_INFO, "You are CONSUMER.");

        pthread_create(&tid, NULL, consumer_thread, new int(sock));
        pthread_detach(tid);

        // consumer: main thread čeká jen na ENTER pro ukončení
        getchar();
        g_stop = 1;
    }

    else
    {
        log_msg(LOG_ERROR, "Unknown role.");
        close(sock);
        exit(1);
    }

    return 0;
}
