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
#include <semaphore.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <signal.h>
#include <mqueue.h>
#include <sys/stat.h>   // for mode constants

#define STR_QUIT    "quit"

//***************************************************************************
// Log messages

#define LOG_ERROR   0
#define LOG_INFO    1
#define LOG_DEBUG   2

int g_debug = LOG_INFO;

int sequence_number = 0;

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
// POSIX named IPC: shared memory + named semaphores OR POSIX message queue
// - support two modes selected via command line: -shm or -mq
// - use shm_open / ftruncate / mmap for shared memory
// - use sem_open for named semaphores
// - use mq_open for POSIX message queue
//
// Keep track whether this process created each IPC object so we unlink only when appropriate.

#define N 100
#define MAX_STR 50

#define SHM_NAME "/osy_hw5_shm"
#define SEM_MUTEX_NAME "/osy_hw5_mutex"
#define SEM_EMPTY_NAME "/osy_hw5_empty"
#define SEM_FULL_NAME  "/osy_hw5_full"
#define MQ_NAME "/osy_hw5_mq"
#define SEM_SEND_NAME "/osy_hw5_send"

// Shared buffer structure (no pointers)
struct shared_area {
    int magic;                 // simple magic/initialized flag
    int in;
    int out;
    char buffer[N][MAX_STR];
};

shared_area *sh = NULL;
int shm_fd = -1;
bool created_shm = false;

sem_t *g_sem_mutex = NULL;
sem_t *g_sem_empty = NULL;
sem_t *g_sem_full = NULL;
// stop 
sem_t *g_sem_send = NULL;
bool created_sem_mutex = false;
bool created_sem_empty = false;
bool created_sem_full = false;
bool created_sem_send = false;

mqd_t g_mqd = (mqd_t)-1;
bool created_mq = false;

int g_use_mq = 0;

//***************************************************************************
// helpers to open/create named semaphores and mq/shm safely

sem_t* open_named_sem(const char* name, unsigned int initval, bool &created)
{
    sem_t *s = sem_open(name, O_CREAT | O_EXCL, 0666, initval);
    if (s != SEM_FAILED) { created = true; return s; }
    if (errno == EEXIST) {
        s = sem_open(name, 0);
        if (s == SEM_FAILED) {
            log_msg(LOG_ERROR, "sem_open(%s) existing failed", name);
            return NULL;
        }
        created = false;
        return s;
    }
    log_msg(LOG_ERROR, "sem_open(%s) failed", name);
    return NULL;
}

mqd_t open_named_mq(const char* name, struct mq_attr *attr, bool &created)
{
    mqd_t q = mq_open(name, O_CREAT | O_EXCL | O_RDWR, 0666, attr);
    if (q != (mqd_t)-1) { created = true; return q; }
    if (errno == EEXIST) {
        q = mq_open(name, O_RDWR);
        if (q == (mqd_t)-1) { log_msg(LOG_ERROR, "mq_open existing failed"); return (mqd_t)-1; }
        created = false;
        return q;
    }
    log_msg(LOG_ERROR, "mq_open failed");
    return (mqd_t)-1;
}

//***************************************************************************
// Abstraction for inserting/removing items

void insert_item(const char* item)
{
    char numbered[MAX_STR];
    snprintf(numbered, sizeof(numbered), "%d. %s", sequence_number++, item);
        if (g_use_mq) {
        if (mq_send(g_mqd, numbered, strlen(numbered) + 1, 0) == -1) {
            log_msg(LOG_ERROR, "mq_send failed");
        }
    } else {
        if (!sh || !g_sem_empty || !g_sem_mutex || !g_sem_full) {
            log_msg(LOG_ERROR, "insert_item: IPC not initialized");
            return;
        }
        sem_wait(g_sem_empty);
        sem_wait(g_sem_mutex);

        strncpy(sh->buffer[sh->in], numbered, MAX_STR-1);
        sh->buffer[sh->in][MAX_STR-1] = 0;
        sh->in = (sh->in + 1) % N;

        sem_post(g_sem_mutex);
        sem_post(g_sem_full);
    }
}

void remove_item(char* item)
{
    if (g_use_mq) {
        ssize_t r = mq_receive(g_mqd, item, MAX_STR, NULL);
        if (r == -1) {
            log_msg(LOG_ERROR, "mq_receive failed");
            item[0] = 0;
        } else {
            if (r >= MAX_STR) item[MAX_STR-1] = 0; else item[r] = 0;
        }
    } else {
        if (!sh || !g_sem_empty || !g_sem_mutex || !g_sem_full) {
            log_msg(LOG_ERROR, "remove_item: IPC not initialized");
            item[0] = 0;
            return;
        }
        sem_wait(g_sem_full);
        sem_wait(g_sem_mutex);

        strncpy(item, sh->buffer[sh->out], MAX_STR-1);
        item[MAX_STR-1] = 0;
        sh->out = (sh->out + 1) % N;

        sem_post(g_sem_mutex);
        sem_post(g_sem_empty);
    }
}

//***************************************************************************
// Per-client process handlers (same as before)

void handle_producer_process(int sock)
{
    char buf[MAX_STR];
    while (1)
    {
        int len = read(sock, buf, sizeof(buf)-1);
        if (len <= 0) break;
        buf[len] = 0;
        buf[strcspn(buf, "\r\n")] = 0;

        if(buf[0] == '-'){
            if (g_sem_send) {
                sem_trywait(g_sem_send); // ignore failure (already paused)
            }
            write(sock, "OK\n", 3);
            printf("Producer requested PAUSE\n");
            continue;
        }
        if (buf[0] == '+'){
            if (g_sem_send) {
                int sval = 0;
                sem_getvalue(g_sem_send, &sval);
                if (sval == 0) sem_post(g_sem_send); // only post if paused
            }
            write(sock, "OK\n", 3);
            printf("Producer requested RESUME\n");
            continue;
        }

        // normal item: respect global send-gate (blocks here if paused)
        if (g_sem_send) {
            sem_wait(g_sem_send);
            sem_post(g_sem_send);
        }

        insert_item(buf);
        write(sock, "OK\n", 3);
        printf("Produced: %s -> OK\n", buf);
    }
    close(sock);
    _exit(0);
}

void handle_consumer_process(int sock)
{
    char item[MAX_STR];
    char sendbuf[MAX_STR + 5];
    while (1)
    {
        remove_item(item);
        int msg_len = snprintf(sendbuf, sizeof(sendbuf), "%s\n", item);
        write(sock, sendbuf, msg_len);

        char okbuf[16];
        int len = read(sock, okbuf, sizeof(okbuf));
        if (len <= 0) break;
    }
    close(sock);
    _exit(0);
}

//***************************************************************************
// Help

void help( int t_narg, char **t_args )
{
    if ( t_narg <= 1 || !strcmp( t_args[ 1 ], "-h" ) )
    {
        printf(
            "\n"
            "  Producer/Consumer Task Server (process-per-client)\n"
            "\n"
            "  Use: %s [-h -d] -shm|-mq port_number\n"
            "\n"
            "    -d    debug mode\n"
            "    -h    this help\n"
            "    -shm  use shared-memory queue implementation\n"
            "    -mq   use POSIX message queue implementation\n"
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
    int flag_shm = 0, flag_mq = 0;

    for ( int i = 1; i < t_narg; i++ )
    {
        if (!strcmp(t_args[i], "-d")) g_debug = LOG_DEBUG;
        if (!strcmp(t_args[i], "-h")) help(t_narg, t_args);
        if (!strcmp(t_args[i], "-shm")) flag_shm = 1;
        if (!strcmp(t_args[i], "-mq"))  flag_mq  = 1;

        if (*t_args[i] != '-' && !l_port)
            l_port = atoi(t_args[i]);
    }

    if (flag_shm + flag_mq != 1) {
        log_msg(LOG_ERROR, "Specify exactly one of -shm or -mq");
        help(t_narg, t_args);
    }
    g_use_mq = flag_mq;

    if (l_port <= 0)
    {
        log_msg(LOG_INFO, "Missing or bad port!");
        help(t_narg, t_args);
    }

    // Initialize IPC depending on mode
    if (g_use_mq) {
        struct mq_attr attr;
        attr.mq_flags = 0;
        attr.mq_maxmsg = N;        // may be capped by system
        attr.mq_msgsize = MAX_STR; // message length
        attr.mq_curmsgs = 0;

        // try create-exclusive, otherwise open existing
        long sys_maxmsg = 0, sys_msgsize_max = 0;
        FILE *f = fopen("/proc/sys/fs/mqueue/msg_max", "r");
        if (f) { if (fscanf(f, "%ld", &sys_maxmsg) != 1) sys_maxmsg = 0; fclose(f); }
        f = fopen("/proc/sys/fs/mqueue/msgsize_max", "r");
        if (f) { if (fscanf(f, "%ld", &sys_msgsize_max) != 1) sys_msgsize_max = 0; fclose(f); }

        if (sys_maxmsg > 0 && attr.mq_maxmsg > sys_maxmsg) attr.mq_maxmsg = (long)sys_maxmsg;
        if (sys_msgsize_max > 0 && attr.mq_msgsize > sys_msgsize_max) {
            attr.mq_msgsize = (long)sys_msgsize_max;
            if (attr.mq_msgsize <= 0) attr.mq_msgsize = 1;
        }

        // try create-exclusive, otherwise open existing
        mq_unlink(MQ_NAME); // best-effort cleanup before create (optional)
        g_mqd = open_named_mq(MQ_NAME, &attr, created_mq);
        if (g_mqd == (mqd_t)-1 && errno == EINVAL) {
            // fallback: try smaller queue depth if system limit info not available
            log_msg(LOG_DEBUG, "mq_open returned EINVAL; retrying with smaller mq_maxmsg");
            attr.mq_maxmsg = (sys_maxmsg > 0) ? sys_maxmsg : 8;
            if (attr.mq_maxmsg <= 0) attr.mq_maxmsg = 8;
            g_mqd = open_named_mq(MQ_NAME, &attr, created_mq);
        }
        if (g_mqd == (mqd_t)-1) {
            log_msg(LOG_ERROR, "mq_open final failed");
            exit(1);
        }

        // create/open global send-gate semaphore (starts unblocked)
        g_sem_send = open_named_sem(SEM_SEND_NAME, 1, created_sem_send);
        if (!g_sem_send) {
            log_msg(LOG_ERROR, "Failed to open send-gate semaphore");
            if (g_mqd != (mqd_t)-1) { mq_close(g_mqd); if (created_mq) mq_unlink(MQ_NAME); }
            exit(1);
        }

    } else {
        // shared memory: try create-excl, else open existing
        shm_fd = shm_open(SHM_NAME, O_CREAT | O_EXCL | O_RDWR, 0666);
        if (shm_fd != -1) {
            // created new
            created_shm = true;
            if (ftruncate(shm_fd, sizeof(shared_area)) == -1) {
                log_msg(LOG_ERROR, "ftruncate failed");
                close(shm_fd);
                shm_unlink(SHM_NAME);
                exit(1);
            }
        } else if (errno == EEXIST) {
            shm_fd = shm_open(SHM_NAME, O_RDWR, 0);
            if (shm_fd == -1) {
                log_msg(LOG_ERROR, "shm_open existing failed");
                exit(1);
            }
            created_shm = false;
        } else {
            log_msg(LOG_ERROR, "shm_open failed");
            exit(1);
        }

        sh = (shared_area*)mmap(NULL, sizeof(shared_area),
                                PROT_READ | PROT_WRITE,
                                MAP_SHARED, shm_fd, 0);
        if (sh == MAP_FAILED) {
            log_msg(LOG_ERROR, "mmap failed");
            if (created_shm) { shm_unlink(SHM_NAME); }
            close(shm_fd);
            exit(1);
        }

        // initialize buffer indices only if we created the shm
        if (created_shm) {
            sh->magic = 0xBEEF;
            sh->in = 0;
            sh->out = 0;
            // clear contents
            for (int i=0;i<N;i++) sh->buffer[i][0]=0;
        } else {
            // optional sanity check: if magic not set, initialize anyway
            if (sh->magic != 0xBEEF) {
                sh->magic = 0xBEEF;
                sh->in = 0;
                sh->out = 0;
                for (int i=0;i<N;i++) sh->buffer[i][0]=0;
            }
        }

        // open/create named semaphores (use O_CREAT|O_EXCL to determine creator)
        g_sem_mutex = open_named_sem(SEM_MUTEX_NAME, 1, created_sem_mutex);
        g_sem_empty = open_named_sem(SEM_EMPTY_NAME, N, created_sem_empty);
        g_sem_full  = open_named_sem(SEM_FULL_NAME, 0, created_sem_full);
        if (!g_sem_mutex || !g_sem_empty || !g_sem_full) {
            log_msg(LOG_ERROR, "sem_open failed");
            // cleanup resources we created
            if (g_sem_mutex) sem_close(g_sem_mutex);
            if (g_sem_empty) sem_close(g_sem_empty);
            if (g_sem_full)  sem_close(g_sem_full);
            if (created_shm) { munmap(sh, sizeof(shared_area)); close(shm_fd); shm_unlink(SHM_NAME); }
            exit(1);
        }

        // create/open global send-gate semaphore (starts unblocked)
        g_sem_send = open_named_sem(SEM_SEND_NAME, 1, created_sem_send);
        if (!g_sem_send) {
            log_msg(LOG_ERROR, "Failed to open send-gate semaphore");
            // cleanup resources we created
            if (g_sem_mutex) sem_close(g_sem_mutex);
            if (g_sem_empty) sem_close(g_sem_empty);
            if (g_sem_full)  sem_close(g_sem_full);
            if (created_shm) { munmap(sh, sizeof(shared_area)); close(shm_fd); shm_unlink(SHM_NAME); }
            exit(1);
        }
    }

    log_msg(LOG_INFO, "Server will listen on port: %d. Mode: %s", l_port, g_use_mq ? "POSIX MQ" : "SHM queue");

    // Create listening socket
    int l_sock_listen = socket(AF_INET, SOCK_STREAM, 0);
    if (l_sock_listen == -1)
    {
        log_msg(LOG_ERROR, "Unable to create socket.");
        // cleanup before exit
        if (g_use_mq) { 
            if (g_mqd != (mqd_t)-1) { 
                mq_close(g_mqd); 
                if (created_mq) mq_unlink(MQ_NAME); 
            } 
        }
        else {
            if (g_sem_mutex) { sem_close(g_sem_mutex); if (created_sem_mutex) sem_unlink(SEM_MUTEX_NAME); }
            if (g_sem_empty) { sem_close(g_sem_empty); if (created_sem_empty) sem_unlink(SEM_EMPTY_NAME); }
            if (g_sem_full)  { sem_close(g_sem_full);  if (created_sem_full)  sem_unlink(SEM_FULL_NAME); }
            if (sh) { munmap(sh, sizeof(shared_area)); close(shm_fd); if (created_shm) shm_unlink(SHM_NAME); }
        }
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
        // cleanup
        close(l_sock_listen);
        if (g_use_mq) { 
            if (g_mqd != (mqd_t)-1) { 
                mq_close(g_mqd); 
                if (created_mq) mq_unlink(MQ_NAME); } }
        else {
            if (g_sem_mutex) { sem_close(g_sem_mutex); if (created_sem_mutex) sem_unlink(SEM_MUTEX_NAME); }
            if (g_sem_empty) { sem_close(g_sem_empty); if (created_sem_empty) sem_unlink(SEM_EMPTY_NAME); }
            if (g_sem_full)  { sem_close(g_sem_full);  if (created_sem_full)  sem_unlink(SEM_FULL_NAME); }
            if (sh) { munmap(sh, sizeof(shared_area)); close(shm_fd); if (created_shm) shm_unlink(SHM_NAME); }
        }
        exit(1);
    }

    if (listen(l_sock_listen, 10) < 0)
    {
        log_msg(LOG_ERROR, "Listen failed!");
        // cleanup
        close(l_sock_listen);
        if (g_use_mq) { if (g_mqd != (mqd_t)-1) { mq_close(g_mqd); if (created_mq) mq_unlink(MQ_NAME); } }
        else {
            if (g_sem_mutex) { sem_close(g_sem_mutex); if (created_sem_mutex) sem_unlink(SEM_MUTEX_NAME); }
            if (g_sem_empty) { sem_close(g_sem_empty); if (created_sem_empty) sem_unlink(SEM_EMPTY_NAME); }
            if (g_sem_full)  { sem_close(g_sem_full);  if (created_sem_full)  sem_unlink(SEM_FULL_NAME); }
            if (sh) { munmap(sh, sizeof(shared_area)); close(shm_fd); if (created_shm) shm_unlink(SHM_NAME); }
        }
        exit(1);
    }

    log_msg(LOG_INFO, "Enter 'quit' to quit server.");

    // Poll stdin + listen socket
    pollfd pfds[2];
    pfds[0].fd = STDIN_FILENO; pfds[0].events = POLLIN;
    pfds[1].fd = l_sock_listen; pfds[1].events = POLLIN;

    // Reap children explicitly
    signal(SIGCHLD, SIG_IGN);

    // ===================== MAIN SERVER LOOP ======================
    while (1)
    {
        if (poll(pfds, 2, -1) < 0)
        {
            log_msg(LOG_ERROR, "Poll failed!");
            break;
        }

        // Quit on stdin
        if (pfds[0].revents & POLLIN)
        {
            char buf[128];
            int len = read(STDIN_FILENO, buf, sizeof(buf));
            if (len > 0 && !strncmp(buf, STR_QUIT, strlen(STR_QUIT)))
            {
                log_msg(LOG_INFO, "Server shutting down.");
                break;
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

            pid_t pid = fork();
            if (pid < 0) {
                log_msg(LOG_ERROR, "fork failed");
                close(client);
                continue;
            } else if (pid == 0) {
                // Child process: close listening socket and handle client
                close(l_sock_listen);
                if (!strcasecmp(taskbuf, "producer"))
                    handle_producer_process(client);
                else if (!strcasecmp(taskbuf, "consumer"))
                    handle_consumer_process(client);
                else {
                    write(client, "Unknown role\n", 13);
                    close(client);
                    _exit(0);
                }
            } else {
                // Parent
                close(client);
                while (waitpid(-1, NULL, WNOHANG) > 0) { }
                if (!strcasecmp(taskbuf, "producer"))
                    log_msg(LOG_INFO, "New PRODUCER connected (pid %d).", pid);
                else if (!strcasecmp(taskbuf, "consumer"))
                    log_msg(LOG_INFO, "New CONSUMER connected (pid %d).", pid);
            }
        }
    }

    // Shutdown and cleanup (only unlink objects created by this process)
    close(l_sock_listen);

    if (g_use_mq) {
        if (g_mqd != (mqd_t)-1) mq_close(g_mqd);
        if (created_mq) mq_unlink(MQ_NAME);
        if (g_sem_send) { sem_close(g_sem_send); if (created_sem_send) sem_unlink(SEM_SEND_NAME); }
    } else {
        if (g_sem_mutex) { sem_close(g_sem_mutex); if (created_sem_mutex) sem_unlink(SEM_MUTEX_NAME); }
        if (g_sem_empty) { sem_close(g_sem_empty); if (created_sem_empty) sem_unlink(SEM_EMPTY_NAME); }
        if (g_sem_full)  { sem_close(g_sem_full);  if (created_sem_full)  sem_unlink(SEM_FULL_NAME); }
        if (g_sem_send) { sem_close(g_sem_send); if (created_sem_send) sem_unlink(SEM_SEND_NAME); }
        if (sh) { munmap(sh, sizeof(shared_area)); close(shm_fd); if (created_shm) shm_unlink(SHM_NAME); }
    }

    return 0;
}
