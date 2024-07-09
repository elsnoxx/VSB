#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <sys/param.h>
#include <pthread.h>
#include <cstring>

#define TYPE int

class TaskPart
{
public:
    int m_id;
    int m_from, m_length;
    TYPE *m_data;
    TYPE *m_sorted_data;

    TaskPart(int t_myid, int t_from, int t_length, TYPE *t_data) : m_id(t_myid), m_from(t_from), m_length(t_length), m_data(t_data)
    {
        m_sorted_data = new TYPE[m_length];
    }

    ~TaskPart()
    {
        delete[] m_sorted_data;
    }

    void insertion_sort()
    {
        for (int i = 1; i < m_length; ++i)
        {
            TYPE key = m_data[i];
            int j = i - 1;

            while (j >= 0 && m_data[j] > key)
            {
                m_data[j + 1] = m_data[j];
                --j;
            }
            m_data[j + 1] = key;
        }
    }

    void sort_data()
    {
        insertion_sort();
        for (int i = 0; i < m_length; ++i)
        {
            m_sorted_data[i] = m_data[i];
        }
    }
};


void *my_thread(void *t_void_arg)
{
    TaskPart *lp_task = (TaskPart *)t_void_arg;

    // printf("Thread %d started from %d with length %d...\n", lp_task->m_id, lp_task->m_from, lp_task->m_length);

    lp_task->sort_data();

    // printf("Sorted data in thread %d.\n", lp_task->m_id);

    return NULL;
}

void merge(TYPE *arr, int l, int m, int r)
{
    int n1 = m - l + 1;
    int n2 = r - m;

    TYPE *L = new TYPE[n1];
    TYPE *R = new TYPE[n2];

    for (int i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (int j = 0; j < n2; j++)
        R[j] = arr[m + 1 + j];

    int i = 0, j = 0, k = l;
    while (i < n1 && j < n2)
    {
        if (L[i] <= R[j])
        {
            arr[k] = L[i];
            i++;
        }
        else
        {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    while (i < n1)
    {
        arr[k] = L[i];
        i++;
        k++;
    }

    while (j < n2)
    {
        arr[k] = R[j];
        j++;
        k++;
    }

    delete[] L;
    delete[] R;
}
// merge sort  of array
void merge_sort(TYPE *arr, int l, int r)
{
    if (l >= r)
    {
        return;
    }

    int m = l + (r - l) / 2;
    
    merge_sort(arr, l, m);
    merge_sort(arr, m + 1, r);
    merge(arr, l, m, r);
}

// Time interval between two measurements converted to ms
int timeval_diff_to_ms(timeval *t_before, timeval *t_after)
{
    timeval l_res;
    timersub(t_after, t_before, &l_res);
    return 1000 * l_res.tv_sec + l_res.tv_usec / 1000;
}

// check of result arrays
void check_array(TYPE *arr1, TYPE *arr2, long l_my_length)
{
    int end = 0;
    printf("\n\n\n");
    for (int i = 0; i < l_my_length; ++i)
    {
        if (arr1[i] != arr2[i])
        {
            end = 1;
            break;
        }
    }
    if (end == 0)
    {
        printf("Arrays are same");
    }
    else
    {
        printf("Arrays are not same");
    }

    printf("\n");
}

// generate random numbers
void generate_random_arrays(TYPE *&l_my_array, TYPE *&l_my_array_copy, long l_my_length)
{
    l_my_array = new TYPE[l_my_length];
    if (!l_my_array)
    {
        printf("Not enough memory for array!\n");
        exit(1);
    }
    l_my_array_copy = new TYPE[l_my_length];
    if (!l_my_array_copy)
    {
        printf("Not enough memory for array copy!\n");
        exit(1);
    }

    srand((long)time(NULL));

    printf("Random numbers generation started...");
    for (long i = 0; i < l_my_length; i++)
    {
        // Generování náhodných čísel v rozsahu od -10^9 do 10^9
        l_my_array[i] = (rand() % (int(2e9 + 1))) - int(1e9);
        l_my_array_copy[i] = l_my_array[i];
        if (!(i % 10)) // Adjust this condition based on your preference
        {
            fflush(stdout);
        }
    }
}
// clean up memoory
void cleanup(pthread_t* thread_ids, TaskPart** task_parts, TYPE* l_my_array, TYPE* l_my_array_copy, int threads) {
    delete[] thread_ids;
    for (int i = 0; i < threads; ++i) {
        delete[] task_parts[i]->m_data;
        delete task_parts[i];
    }
    delete[] task_parts;

    delete[] l_my_array;
    delete[] l_my_array_copy;
}

// print arrays
void print_arrays(TYPE* arr1, TYPE* arr2, long length) {
    printf("\nSorted array:\n");
    for (int i = 0; i < length; ++i) {
        printf("%d ", arr1[i]);
    }
    printf("\n\n\n\n\n\n\n\n");
    for (int i = 0; i < length; ++i) {
        printf("%d ", arr2[i]);
    }
    printf("\n");
}


#define LENGTH_LIMIT 10

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        printf("Usage: %s <number_of_elements> <number_of_threads>\n", argv[0]);
        printf("Example: %s 1000 4\n", argv[0]);
        printf("Specify number of elements, at least %d.\n", LENGTH_LIMIT);
        return 0;
    }

    long l_my_length = atol(argv[1]);
    timeval l_time_before, l_time_after;
    int threads = atoi(argv[2]);
    if (threads < 1 || threads > 6)
    {
        printf("The number of threads must be outside the range from 1 to 6.\n");
        return 0;
    }

    TYPE *l_my_array;
    TYPE *l_my_array_copy;

    // Generate random arrays
    generate_random_arrays(l_my_array, l_my_array_copy, l_my_length);
    // Check if arrays are same copy
    // check_array(l_my_array, l_my_array_copy, l_my_length);

    printf("\nSorting using %d threads...\n", threads);
    pthread_t *thread_ids = new pthread_t[threads];
    TaskPart **task_parts = new TaskPart *[threads];

    int part_length = l_my_length / threads;
    int remain_length = l_my_length % threads;
    int from = 0;

    for (int i = 0; i < threads; ++i)
    {
        int length = part_length;

        if (remain_length > 0 && i != threads - 1)
        {
            length++;
            remain_length--;
        }

        // printf("%d, %d, %d\n", i, from, length);
        TYPE *data_copy = new TYPE[length];
        memcpy(data_copy, l_my_array + from, length * sizeof(TYPE));
        task_parts[i] = new TaskPart(i, from, length, data_copy);
        from += length;
    }

    // Time recording before sort
    gettimeofday(&l_time_before, NULL);

    for (int i = 0; i < threads; ++i)
    {
        pthread_create(&thread_ids[i], NULL, my_thread, task_parts[i]);
    }

    for (int i = 0; i < threads; ++i)
    {
        pthread_join(thread_ids[i], NULL);
    }
    // Time recording after sort
    gettimeofday(&l_time_after, NULL);
    printf("The sort time using %d thread: %d [ms]\n", threads, timeval_diff_to_ms(&l_time_before, &l_time_after));

    if (threads != 1)
    {
        gettimeofday(&l_time_before, NULL);

        merge_sort(l_my_array, 0, l_my_length - 1);
        
        gettimeofday(&l_time_after, NULL);
        printf("The Merge time: %d [ms]\n", timeval_diff_to_ms(&l_time_before, &l_time_after));
    }

    printf("\nSorting using 1 thread...\n");
    TaskPart task(1, 0, l_my_length, l_my_array_copy);

    // Starting thread
    pthread_t thread_id;
    gettimeofday(&l_time_before, NULL);
    pthread_create(&thread_id, NULL, my_thread, &task);

    // Waiting for thread completion
    pthread_join(thread_id, NULL);
    gettimeofday(&l_time_after, NULL);

    printf("The sort time using 1 thread: %d [ms]\n", timeval_diff_to_ms(&l_time_before, &l_time_after));

    // print_arrays(l_my_array, l_my_array_copy, l_my_length);

    check_array(l_my_array, l_my_array_copy, l_my_length);

    cleanup(thread_ids, task_parts, l_my_array, l_my_array_copy, threads);

    return 0;
}
