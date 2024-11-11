#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <vector>
#include <algorithm>

using namespace std;

struct Node {
    string name;
    vector<string> passage;

    Node() : name(""), passage({}) {}
    Node(string name, vector<string> passage) : name(name), passage(passage) {}
};

void dfs(const string& currentCave, vector<string>& path, map<string, Node>& jeskyne, vector<vector<string>>& allPaths) {
    path.push_back(currentCave);

    if (currentCave == "end") {
        allPaths.push_back(path);
    }
    else {
        for (const string& neighbor : jeskyne[currentCave].passage) {
            if (islower(neighbor[0]) && find(path.begin(), path.end(), neighbor) != path.end()) {
                continue;
            }
            dfs(neighbor, path, jeskyne, allPaths);
        }
    }

    path.pop_back();
}

int main() {
    vector<string> FilesName = { "..\\Data\\test0.txt", "..\\Data\\test1.txt", "..\\Data\\test2.txt", "..\\Data\\test3.txt", "..\\Data\\trivial_test.txt"};

    for (int i = 0; i < FilesName.size(); i++) {
        ifstream file(FilesName[i]);
        if (!file) {
            cerr << "Nepodařilo se otevřít soubor: " << FilesName[i] << endl;
            return 1;
        }

        map<string, Node> jeskyne;
        string myText;

        while (getline(file, myText)) {
            size_t positionOfDash = myText.find("-");

            string jeskyn1 = myText.substr(0, positionOfDash);
            string jeskyn2 = myText.substr(positionOfDash + 1);

            jeskyne[jeskyn1].name = jeskyn1;
            jeskyne[jeskyn1].passage.push_back(jeskyn2);

            jeskyne[jeskyn2].name = jeskyn2;
            jeskyne[jeskyn2].passage.push_back(jeskyn1);
        }


        vector<vector<string>> allPaths;
        vector<string> path;
        dfs("start", path, jeskyne, allPaths);

        cout << FilesName[i] << ": " << allPaths.size() << endl;

        /*
        for (const auto& p : allPaths) {
            for (const auto& cave : p) {
                cout << cave << " ";
            }
            cout << endl;
        }
        cout << endl;
        */
    }

    return 0;
}

