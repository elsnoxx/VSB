#include "FileUtil.h"

bool file_can_execute(const char *filename) {
    return access(filename, X_OK) == 0;
}

bool file_can_read(const char *filename) {
    return access(filename, R_OK) == 0;
}

bool is_directory(const char *path) {
    struct stat myStat;
    if (stat(path, &myStat) != 0) {
        return false;
    }
    return S_ISDIR(myStat.st_mode);
}

void getInfo(const char *filename) {
    struct stat myStat;
    if (stat(filename, &myStat) != 0) {
        printf("Chyba při získávání informací o souboru: %s\n", filename);
        return;
    }
    printf("Informace o souboru: %s\n", filename);
    printf("Velikost: %lld bajtů\n", (long long)myStat.st_size);
    printf("Poslední přístup: %s", ctime(&myStat.st_atime));
    printf("Poslední změna: %s", ctime(&myStat.st_mtime));
}