/**
 * @brief Main function that processes files and finds paths in a cave system.
 *
 * This function loads cave system data from text files and, for each file,
 * finds all possible paths using the PathFinder. The results are printed to the console.
 *
 * @return Returns 0 if all files are successfully processed, or 1 if an error occurs.
 */
#include <iostream>
#include <vector>
#include <string>
#include "CaveSystem.h"
#include "PathFinder.h"

int main() {
    std::vector<std::string> files = { "..\\Data\\test0.txt", "..\\Data\\test1.txt", "..\\Data\\test2.txt",
                                       "..\\Data\\test3.txt", "..\\Data\\trivial_test.txt" };

    for (const auto& filename : files) {
        try {
            CaveSystem caveSystem;
            caveSystem.loadFromFile(filename);

            PathFinder pathFinder;
            std::vector<std::vector<std::string>> allPaths = pathFinder.findAllPaths(caveSystem);

            std::cout << filename << ": " << allPaths.size() << " paths found." << std::endl;

            // Vypiš všechny nalezené cesty
            //pathFinder.printAllPaths();

        }
        catch (const std::exception& e) {
            std::cerr << e.what() << std::endl;
            return 1;
        }
    }

    return 0;
}

