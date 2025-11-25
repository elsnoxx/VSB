#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>

#define N 5
#define MAX_STR 50
#define PRODUCE_COUNT 10

char buffer[N][MAX_STR];
int in = 0;
int out = 0;

// 3 semafory
sem_t empty;
sem_t full;
sem_t mutex;

void put_item(const char *item)
{
    sem_wait(&empty);
    sem_wait(&mutex);
    strncpy(buffer[in], item, MAX_STR - 1);
    buffer[in][MAX_STR - 1] = '\0';
    in = (in + 1) % N;
    printf("Produced: %s\n", item);
    sem_post(&mutex);
    sem_post(&full);
}

void get_item(char *item)
{
    sem_wait(&full);
    sem_wait(&mutex);
    strncpy(item, buffer[out], MAX_STR - 1);
    item[MAX_STR - 1] = '\0';
    out = (out + 1) % N;
    sem_post(&mutex);
    sem_post(&empty);
}

void *producer_thread(void *arg)
{
    for (int i = 0; i < PRODUCE_COUNT; ++i)
    {
        char msg[MAX_STR];
        snprintf(msg, sizeof(msg), "Message %d", i);
        put_item(msg);
        usleep(100000);
    }
    return nullptr;
}

void *consumer_thread(void *arg)
{
    char item[MAX_STR];
    for (int i = 0; i < PRODUCE_COUNT; ++i)
    {
        get_item(item);
        printf("Consumed: %s\n", item);
        usleep(150000);
    }
    return nullptr;
}

int main()
{
    if (sem_init(&mutex, 0, 1) != 0)
    {
        perror("sem_init mutex");
        return 1;
    }
    if (sem_init(&empty, 0, N) != 0)
    {
        perror("sem_init empty");
        return 1;
    }
    if (sem_init(&full, 0, 0) != 0)
    {
        perror("sem_init full");
        return 1;
    }

    pthread_t prod, cons;

    if (pthread_create(&prod, NULL, producer_thread, NULL) != 0)
    {
        perror("pthread_create prod");
        return 1;
    }
    if (pthread_create(&cons, NULL, consumer_thread, NULL) != 0)
    {
        perror("pthread_create cons");
        return 1;
    }
    pthread_join(prod, NULL);
    pthread_join(cons, NULL);
    sem_destroy(&mutex);
    sem_destroy(&empty);
    sem_destroy(&full);
    return 0;
}