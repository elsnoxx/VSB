#include "maytail.h"

//main test function
int main(int argc, char *argv[]) {
    vector<string> files;
    const char* logfile;

    parse_args(argc, argv, files, logfile);

    if (!logfile) {
        fprintf(stderr, "Chyba: nebyl zadán logovací soubor pomocí -l\n");
        printHelp();
        return 1;
    }

    if (files.empty()) {
        printHelp();
        return 1;
    }

    getFileInfo(files, logfile);

    // printFiles(files);

    return 0;
}