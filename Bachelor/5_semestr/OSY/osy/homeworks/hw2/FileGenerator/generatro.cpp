#include <stdlib.h>
#include <stdio.h>
#include <cstring>
#include <iostream>
#include <fstream>
#include <unistd.h>

using namespace std;

void help(){
    printf("./generatro [options]\n");
    printf("Options:\n");
    printf("  -h, --help       Show this help message\n");
    printf("  -d, --delay      <delay_time>\n");
    printf("  -o, --output     <output_file>\n");
    printf("  -n, --number     <number_of_items> or default 10000\n");
}

void generate(int delay_time, const char* output_file, int number_of_items) {
    printf("Generating something...\n");

    ofstream myfile;
    if (output_file) {
        myfile.open(output_file, ios::app);
    }
    for (int i = 0; i < number_of_items; i++) {
        // Simulate some processing delay
        if (delay_time > 0) {
            usleep(delay_time * 1000); // Convert ms to us
        }
        if (output_file) {
            myfile << "Item " << i + 1 << "\n";
        } else {
            printf("Item %d\n", i + 1);
        }
    }

    
    myfile.close();
    

    printf("Generation complete. Generated %d items.\n", number_of_items);
    if (output_file) {
        printf("Output written to %s\n", output_file);
    } else {
        printf("No output file specified.\n");
    }
}

void parse_args(int argc, char **argv, int &delay_time, char* &output_file, int &number_of_items) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            help();
            break;
        } else if (strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--delay") == 0) {
            if (i + 1 < argc) {
                delay_time = atoi(argv[++i]);
                printf("Delay time set to %d\n", delay_time);
            } else {
                fprintf(stderr, "Error: No delay time provided\n");
                break;
            }
        } else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) {
            if (i + 1 < argc) {
                output_file = argv[++i];
                printf("Output file set to %s\n", output_file);
            } else {
                fprintf(stderr, "Error: No output file provided\n");
                break;
            }
        } else if (strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--number") == 0) {
            if (i + 1 < argc) {
                number_of_items = atoi(argv[++i]);
                printf("Number of items set to %d\n", number_of_items);
            } else {
                fprintf(stderr, "Error: No number of items provided\n");
                break;
            }
        }
    }
}

int main(int argc, char **argv) {
    int delay_time = 0;
    char *output_file = NULL;
    int number_of_items = 10000;

    parse_args(argc, argv, delay_time, output_file, number_of_items);

    if (argc == 1) {
        help();
        return 0;
    } else if (output_file == NULL && delay_time == 0 && number_of_items == 10000) {
        fprintf(stderr, "Error: No valid options provided. Use -h or --help for usage information.\n");
        return 1;
    }

    printf("Final settings:\n");
    printf("  Delay time: %d\n", delay_time);
    printf("  Output file: %s\n", output_file ? output_file : "None");
    printf("  Number of items: %d\n", number_of_items);

    generate(delay_time, output_file, number_of_items);

    return 0;
}