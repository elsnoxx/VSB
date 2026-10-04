#ifndef PARSE_ARGS_H
#define PARSE_ARGS_H

#include <vector>
#include <string>

#include "FileUtil.h"

using namespace std;

// Deklarace funkcí

// vypis nápovědy
void printHelp();

// vypis souborů
void printFiles(const vector<string> &files);

// parsování argumentů
void parse_args(int argc, char *argv[], vector<string> &files);


#endif // PARSE_ARGS_H