#ifndef MONITOR_H
#define MONITOR_H

#include <vector>
#include <string>
#include <sys/wait.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>

#include "FileUtil.h"

using namespace std;

// Deklarace funkcí

// monitorování souborů
void getFileInfo(const vector<string> &files, const char* logFileName);

#endif