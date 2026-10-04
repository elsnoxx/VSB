# Operační systémy - VSB
<a href="https://poli.cs.vsb.cz/edu/osy" target="_blank">https://poli.cs.vsb.cz/edu/osy</a>

## Snippets
- [fork()](snippets/fork.md)
- [Pipes](snippets/pipe.md)
- [dup2()](snippets/dup2.md)
- [exec()](snippets/exec.md)
- [grep](snippets/grep.md)
- [stat()](snippets/stat.md)
- [Polling](snippets/poll.md)
- [Sockets](snippets/socket.md)
- [Signals](snippets/signal.md)
- [Threads](snippets/thread.md)
- [Semaphores](snippets/semaphore.md)
- [Shared memory](snippets/shared_memory.md)
- [Message_queue](snippets/msg_queue.md)
- [String operations](snippets/string_operations.md)
- [File operations](snippets/file_operations.md)
- [Makefile](snippets/makefile.md)

# Poznámky

# Dobré funkce
- **strtok** -- splitne string, podle mezery tabulatoru a enteru, pak se musi cyklem projit a hledat
```c++
int arg_count = 0;
char *args[10]; // assume max number of arguments

char *token = strtok(buffer, " \t\n");
while (token != NULL && arg_count < 10)
{
    args[arg_count++] = token;
    token = strtok(NULL, " \t\n");
}
```

- **exec** -- spousti program an rhazuje tim aktualni
    - **l**	list – argumenty jako seznam (arg1, arg2, ..., NULL)
    - **v**	vector – argumenty jako pole (char *argv[])
    - **p**	path – hledá program v $PATH
    - **e**	environment – explicitní envp[]


## Funkce Poll
Není potřebná jelikož v zadání tvoříme pro klinta vždy nový proces a nebo thread. To zapříčiní smazaní zbytečného kódu a nahrazení za jednodušší.

```c++

void handle_client_fork(int client_socket, int index);

void *client_handler_thread(void *arg)
{
	int client_socket = *(int *)arg;
	char buffer[256];
	int bytes_read = read(client_socket, buffer, sizeof(buffer));
}

while ( 1 )
{
    struct sockaddr_in client_addr;
    socklen_t client_len = sizeof(client_addr);
    int client_socket = accept(listen_socket, (struct sockaddr *)&client_addr, &client_len);
    if (client_socket < 0)
    {
        log_msg(LOG_ERROR, "Accept failed!");
        continue;
    }

    // thread creation for client handling would be here
    pthread_t client_thread;
    if (pthread_create(&thread, NULL, client_handler_thread, (void *)&l_sock_client) != 0)
    {
        log_msg(LOG_ERROR, "Failed to create thread");
        exit(1);
    }

    // fork proces pro klienrata by byl zde
    pid_t pid = fork();
    if (pid < 0) {
        log_msg(LOG_ERROR, "Fork failed!");
        close(client_socket);
    }
    else if (pid == 0){
        // child procces
        close(l_sock_listen);
        handle_client_fork(l_sock_client);
        
    } else if (pid > 0){
        // parent process
        close(client_socket);
        // optionally wait for child processes to prevent zombies
        //waitpid(-1, NULL, WNOHANG);
    }

}
```

## Semafory

- DOWN
```c++
sem_wait()
```

- UP
```c++
sem_post()
```

- Pokud používáš thready
```c++
sem_t sem;
sem_init(&sem, 0, 1);
sem_wait(&sem);
sem_post(&sem);
sem_destroy(&sem);
```

- Pokud používáš procesy
```c++
sem_t *sem = sem_open("/sem", O_CREAT, 0666, 1);
sem_wait(sem);
sem_post(sem);
sem_close(sem);
sem_unlink("/sem");
```