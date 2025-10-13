#include "ParseArgs.h"

void parse_args(int argc, char *argv[], vector<string> &files) {
    for (int i = 1; i < argc; i++) {
        if (is_directory(argv[i]) || file_can_execute(argv[i]) || !file_can_read(argv[i])) {
            continue;
        } else {
            files.push_back(argv[i]);
        }
    }
}

void printHelp(){
    printf("Usage: maytail [options] [files...]\n");
}

void printFiles(const vector<string> &files) {
    for (size_t i = 0; i < files.size(); i++) {
        printf("File %zu: %s\n", i + 1, files[i].c_str());
    }
}