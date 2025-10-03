#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

void genereteBankNumber(long N, long S, bool binary) {
    for (long i = 0; i < N; i++) {
        long val = S + i;
        if (binary) {
            write(1, &val, sizeof(val));
        }
        else {
            printf("%010ld\n", val);
        }
    }
}