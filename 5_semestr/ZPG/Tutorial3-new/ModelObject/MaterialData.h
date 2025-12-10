#pragma once
#include <glm/glm.hpp>

// Jednoduchá struktura popisující materiál (ambient, diffuse, specular, shininess)
struct MaterialData {
    glm::vec3 ambient;
    glm::vec3 diffuse;
    glm::vec3 specular;
    float shininess;
};