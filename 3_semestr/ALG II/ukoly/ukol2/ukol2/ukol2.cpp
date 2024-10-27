#include <iostream>
#include <map>
#include <vector>
#include <string>
#include <queue>
#include <unordered_set>
#include <algorithm>


struct State {
    std::string name;
    std::vector<std::string> transitions;
    bool isValid;


    State() : name(""), transitions({}), isValid(true) {}


    State(std::string n, std::vector<std::string> t, bool valid = true)
        : name(n), transitions(t), isValid(valid) {}
};

// Hlavní graf
std::map<std::string, State> graf = {
    {"Pkvz ||", State("Pkvz ||", {"kv || Pz", "vz || Pk", "kz || Pv", "kvz || P"}, true)},
    {"kv || Pz", State("kv || Pz", {"Pkv || z", "Pkvz ||"}, false)},
    {"vz || Pk", State("vz || Pk", {"Pvz || k", "Pkvz ||"}, true)},
    {"kz || Pv", State("kz || Pv", {"Pkz || v", "Pkvz ||"}, false)},
    {"kvz || P", State("kvz || P", {"Pkvz ||"}, false)},
    {"Pkv || z", State("Pkv || z", {"v || Pkz", "k || Pvz", "kv || Pz"}, true)},
    {"Pvz || k", State("Pvz || k", {"v || Pkz", "z || Pkv", "vz || Pk"}, true)},
    {"v || Pkz", State("v || Pkz", {"Pvz || k", "Pkv || z", "Pv || kz"}, true)},
    {"Pk || vz", State("z || Pkz", {"|| Pkvz", "k || Pvz"}, true)},
    {"Pv || kz", State("Pv || kz", {"v || Pkz", "|| Pkvz"}, false)},
    {"k || Pvz", State("k || Pvz", {"Pk || vz", "Pkz || v", "Pkv || z"}, true)},
    {"Pkz || v", State("Pkz || v", {"z || Pkv", "k || Pvz", "kz || Pv"}, true)},
    {"z || Pkv", State("z || Pkv", {"Pz || kv", "Pkz || v", "Pvz || k"}, true)},
    {"Pz || kv", State("Pz || kv", {"z || Pkv", "|| Pkvz"}, false)},
    {"|| Pkvz", State("|| Pkvz", {"P || kvz", "Pz || kv", "Pk || vz", "Pv || kz"}, true)},
    {"P || kvz", State("P || kvz", {"|| Pkvz"}, true)}
};

// Funkce pro provádìní BFS
std::vector<std::string> bfs(const std::string& start, const std::string& goal) {
    std::queue<std::pair<std::string, std::string>> to_visit;
    std::unordered_set<std::string> visited;
    std::map<std::string, std::string> parent;

    to_visit.push({ start, "" });
    visited.insert(start);

    while (!to_visit.empty()) {
        std::pair<std::string, std::string> front = to_visit.front();
        to_visit.pop();

        std::string current = front.first;
        std::string previous = front.second;


        if (current == goal) {
            std::vector<std::string> path;
            for (std::string at = goal; !at.empty(); at = parent[at]) {
                path.push_back(at);
            }
            std::reverse(path.begin(), path.end());
            return path;
        }

        for (const auto& transition : graf[current].transitions) {
            if (graf.find(transition) != graf.end()) {
                if (!graf[transition].isValid) {
                    continue;
                }

                if (visited.find(transition) == visited.end()) {
                    visited.insert(transition);
                    to_visit.push({ transition, current });
                    parent[transition] = current;
                }
            }
        }
    }
    return {};
}

int main() {
    setlocale(LC_ALL, "cs_CZ");
    std::string start = "Pkvz ||";
    std::string goal = "|| Pkvz";

    std::vector<std::string> path = bfs(start, goal);

    if (!path.empty()) {
        std::cout << "Nalezena cesta: ";
        std::cout << "\n\n";
        for (const auto& state : path) {
            std::cout << state << "\n";
        }
        std::cout << std::endl;
    }
    else {
        std::cout << "Cesta nenalezena." << std::endl;
    }

    return 0;
}
