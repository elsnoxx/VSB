#include "MaterialManager.h"

// If your project provides a free function `getMaterial(MaterialType)` in another
// translation unit, it can be declared `extern` and used here. Otherwise this
// manager supplies a set of predefined materials and returns pointers to
// `MaterialData` instances below.
extern const MaterialData* getMaterial(MaterialType mt);

const MaterialData* MaterialManager::get(MaterialType mt) const {
    // Predefined materials used by scenes.
    // Format: ambient, diffuse, specular, shininess [, emissive]
    static const MaterialData matWood{ {0.2f,0.12f,0.05f}, {0.6f,0.4f,0.2f},   {0.05f,0.05f,0.05f},  16.0f }; // wood
    static const MaterialData matPlastic{ {0.1f,0.1f,0.1f},   {0.5f,0.5f,0.5f},   {0.3f,0.3f,0.3f},    32.0f }; // plastic
    static const MaterialData matMetal{ {0.25f,0.25f,0.25f},{0.4f,0.4f,0.4f},   {0.8f,0.8f,0.8f},    64.0f }; // metal
    static const MaterialData matConst{ {1.0f,1.0f,1.0f},   {1.0f,1.0f,1.0f},   {0.0f,0.0f,0.0f},    1.0f }; // constant color
    static const MaterialData matSky{ {0.3f,0.35f,0.45f}, {0.3f,0.35f,0.45f}, {0.0f,0.0f,0.0f},    1.0f }; // skydome / unlit
    // Emissive material: not affected by scene lights, adds its own emitted light.
    static const MaterialData matEmissive{ {0,0,0}, {0,0,0}, {0,0,0}, 1.0f ,{3.0f,3.0f,2.0f} };

    

    // Return pointer to corresponding MaterialData for the given MaterialType enum
    switch (mt) {
    case MaterialType::Wood:     return &matWood;
    case MaterialType::Plastic:  return &matPlastic;
    case MaterialType::Metal:    return &matMetal;
    case MaterialType::Constant: return &matConst;
    case MaterialType::Skydome:  return &matSky;
    case MaterialType::Emissive: return &matEmissive;
    default:                     return &matConst;
    }
}