#pragma once
#include <glm/glm.hpp>

// Jednoduch� struktura popisuj�c� materi�l (ambient, diffuse, specular, shininess)
struct MaterialData {
    glm::vec3 ambient;
    glm::vec3 diffuse;
    glm::vec3 specular;
    float shininess;
    glm::vec3 emissive = glm::vec3(0.0f);
};