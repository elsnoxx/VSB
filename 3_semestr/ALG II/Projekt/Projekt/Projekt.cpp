#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <vector>

using namespace std;

struct Node {
    string name;
    vector<string> passage;

    Node() : name(""), passage({}) {}
    Node(string name, vector<string> passage) : name(name), passage(passage) {}
};

int main() {
    // Otev?ení souboru
    ifstream file("C:\\Users\\admin\\Documents\\GitHub\\VSB\\3_semestr\\ALG II\\Projekt\\Data\\test0.txt");
    if (!file) {
        cerr << "Nepoda?ilo se otev?ít soubor." << endl;
        return 1;
    }

    map<string, Node> jeskyne;
    string myText;

    while (getline(file, myText)) {
        size_t positionOfDash = myText.find("-");

        string jeskyn1 = myText.substr(0, positionOfDash);
        string jeskyn2 = myText.substr(positionOfDash + 1);

        cout << jeskyn1 << " " << jeskyn2 << endl;

        
        jeskyne[jeskyn1].name = jeskyn1;
        jeskyne[jeskyn1].passage.push_back(jeskyn2);

        jeskyne[jeskyn2].name = jeskyn2;
        jeskyne[jeskyn2].passage.push_back(jeskyn1);
    }


    for (const auto& pair : jeskyne) {
        const string& key = pair.first;
        const Node& node = pair.second;

        cout << "Jeskyne " << key << " ma spojeni s: ";
        for (const string& neighbor : node.passage) {
            cout << neighbor << " ";
        }
        cout << endl;
    }


    return 0;
}
