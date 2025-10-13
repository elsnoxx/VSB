#include "maytail.h"





//main test
int main(int argc, char *argv[]) {
    vector<string> files;

    parse_args(argc, argv, files);

    if (files.empty()) {
        printHelp();
        return 1;
    }

    getFileInfo(files);

    // printFiles(files);

    return 0;
}