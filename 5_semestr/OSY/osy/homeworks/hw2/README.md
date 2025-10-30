

int main(int argc, char *argv[]) {
    vector<string> files;
    const char* logfile = nullptr;

    parse_args(argc, argv, files, logfile);

    

    if (files.empty()) {
        fprintf(stderr, "Chyba: nebyly zadány žádné validní soubory\n");
        printHelp();
        return 1;
    }

    getFileInfo(files, logfile);

    return 0;
}
