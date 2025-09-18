#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Použití: %s S [N] [-b]\n", argv[0]);
        return 1;
    }

    long S = strtol(argv[1], nullptr, 10);
    long N = (argc > 2 && argv[2][0] != '-') ? strtol(argv[2], nullptr, 10) : 1000;
    bool binary = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-b") == 0) binary = true;
    }

    for (long i = 0; i < N; i++) {
        long val = S + i;
        if (binary) {
            fwrite(&val, sizeof(long), 1, stdout);
        }
        else {
            printf("%010ld\n", val);
        }
    }

    return 0;
}
