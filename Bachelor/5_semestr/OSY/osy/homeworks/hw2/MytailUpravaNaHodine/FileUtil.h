#ifndef FileUtil_H
#define FileUtil_H

#include <string>
#include <vector>
#include <sys/stat.h>
#include <unistd.h>
#include <ctime>

// Funkce pro práci se soubory

// Zkontroluje, zda je soubor spustitelný
bool file_can_execute(const char *filename);

// Zkontroluje, zda je soubor čitelný
bool file_can_read(const char *filename);

// Zkontroluje, zda je cesta adresář
bool is_directory(const char *path);

// Získá a vypíše informace o souboru
void getInfo(const char *filename);

#endif