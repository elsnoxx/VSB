#pragma once
#include <glm/glm.hpp>

// Simple structure describing a material (ambient, diffuse, specular, shininess)
// `emissive` is optional and defaults to zero; non-zero emissive makes the
// material appear self-lit (adds light independent of scene lights).
struct MaterialData {
    glm::vec3 ambient;
    glm::vec3 diffuse;
    glm::vec3 specular;
    float shininess;
    glm::vec3 emissive = glm::vec3(0.0f);
};