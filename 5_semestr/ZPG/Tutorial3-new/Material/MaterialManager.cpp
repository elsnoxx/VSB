#include "MaterialManager.h"

// pokud m�te v projektu volnou funkci getMaterial(MaterialType) definovanou v Materials.cpp,
// deklarujeme ji zde jako extern a pou�ijeme ji.
// Pokud ne, implementujte zde vlastn� mapu MaterialData podle pot�eby.
extern const MaterialData* getMaterial(MaterialType mt);

const MaterialData* MaterialManager::get(MaterialType mt) const {
    static const MaterialData matWood{ {0.2f,0.12f,0.05f}, {0.6f,0.4f,0.2f},   {0.05f,0.05f,0.05f},  16.0f };
    static const MaterialData matPlastic{ {0.1f,0.1f,0.1f},   {0.5f,0.5f,0.5f},   {0.3f,0.3f,0.3f},    32.0f };
    static const MaterialData matMetal{ {0.25f,0.25f,0.25f},{0.4f,0.4f,0.4f},   {0.8f,0.8f,0.8f},    64.0f };
    static const MaterialData matConst{ {1.0f,1.0f,1.0f},   {1.0f,1.0f,1.0f},   {0.0f,0.0f,0.0f},    1.0f };
    static const MaterialData matSky{ {0.3f,0.35f,0.45f}, {0.3f,0.35f,0.45f}, {0.0f,0.0f,0.0f},    1.0f };
    static const MaterialData matEmissive{ {0,0,0}, {0,0,0}, {0,0,0}, 1.0f ,{3.0f,3.0f,2.0f} };

    

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