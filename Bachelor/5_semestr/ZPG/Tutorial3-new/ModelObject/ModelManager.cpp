#include "ModelManager.h"
#include <iostream>

// ModelManager: lazy-loads and caches `Model` instances for the application.
// - Provides a singleton `instance()` accessor.
// - `get(type)` returns a shared_ptr to a cached Model or creates it on demand.
// - `createModel(type)` constructs the Model from either embedded vertex arrays
//   (compiled-in models under ./Models/) or by loading an OBJ file for larger
//   assets. The created `Model` is stored in `cache` and shared with callers.

// include compiled-in model arrays (paths relative to this file)
#include "./Models/sphere.h"
#include "./Models/tree.h"
#include "./Models/bushes.h"
#include "./Models/suzi_flat.h"
#include "./Models/suzi_smooth.h"
#include "./Models/plain.h"
#include "./Models/gift.h"
#include "./Models/skycube.h"
#include "./Models/triangle.h"

namespace {
    inline int countVertices(size_t bytes) {
        // Helper: calculate vertex count from byte size. The embedded model
        // arrays are stored as floats with (position+normal) = 6 floats per vertex
        // for these simple compiled models.
        return static_cast<int>(bytes / (6 * sizeof(float)));
    }
}

ModelManager& ModelManager::instance() {
    static ModelManager inst;
    return inst;
}

std::shared_ptr<Model> ModelManager::get(ModelType type) {
    auto it = cache.find(type);
    if (it != cache.end()) return it->second;
    // Not found in cache -> create, cache and return.
    return createModel(type);
}

std::shared_ptr<Model> ModelManager::createModel(ModelType type) {
    std::shared_ptr<Model> modelPtr;

    switch (type) {
    case ModelType::Triangle:
        modelPtr = std::make_shared<Model>(triangle, sizeof(sphere), countVertices(sizeof(sphere)));
        break;
    case ModelType::Sphere:
        modelPtr = std::make_shared<Model>(sphere, sizeof(sphere), countVertices(sizeof(sphere)));
        break;
    case ModelType::Tree:
        modelPtr = std::make_shared<Model>(tree, sizeof(tree), countVertices(sizeof(tree)));
        break;
    case ModelType::Bushes:
        modelPtr = std::make_shared<Model>(bushes, sizeof(bushes), countVertices(sizeof(bushes)));
        break;
    case ModelType::SuziFlat:
        modelPtr = std::make_shared<Model>(suziFlat, sizeof(suziFlat), countVertices(sizeof(suziFlat)));
        break;
    case ModelType::SuziSmooth:
        modelPtr = std::make_shared<Model>(suziSmooth, sizeof(suziSmooth), countVertices(sizeof(suziSmooth)));
        break;
    case ModelType::Plain:
        modelPtr = std::make_shared<Model>(plain, sizeof(plain), countVertices(sizeof(plain)));
        break;
    case ModelType::Gift:
        modelPtr = std::make_shared<Model>(gift, sizeof(gift), countVertices(sizeof(gift)));
        break;
    case ModelType::Skycube:
        modelPtr = std::make_shared<Model>(skycube, sizeof(skycube), countVertices(sizeof(skycube)));
        break;
    case ModelType::House:
        modelPtr = std::make_shared<Model>("house.obj");
        break;
    case ModelType::Formula1:
        modelPtr = std::make_shared<Model>("formula1.obj");
        break;
    case ModelType::Cube:
        modelPtr = std::make_shared<Model>("cube.obj");
        break;
    case ModelType::Square:
        modelPtr = std::make_shared<Model>("square.obj");
        break;
    case ModelType::Toilet:
        modelPtr = std::make_shared<Model>("toiled.obj");
        break;
    case ModelType::Fiona:
        modelPtr = std::make_shared<Model>("fiona.obj");
        break;
    case ModelType::Shrek:
        modelPtr = std::make_shared<Model>("shrek.obj");
        break;
    case ModelType::Teren:
        modelPtr = std::make_shared<Model>("teren.obj");
        break;
    case ModelType::Venus:
        modelPtr = std::make_shared<Model>("solarSystem/venus/Venus_1K.obj");
        break;
    case ModelType::Earth:
        modelPtr = std::make_shared<Model>("solarSystem/earth/Earth 2K.obj");
        break;
    case ModelType::Mercury:
        modelPtr = std::make_shared<Model>("solarSystem/mercury/Mercury1K.obj");
        break;
    case ModelType::Moon:
        modelPtr = std::make_shared<Model>("solarSystem/moon/Moon2K.obj");
        break;
    case ModelType::Mars:
        modelPtr = std::make_shared<Model>("solarSystem/mars/Mars 2K.obj");
        break;
    case ModelType::Uranus:
        modelPtr = std::make_shared<Model>("solarSystem/urano/13907_Uranus_v2_l3.obj");
        break;
    case ModelType::Pluto:
        modelPtr = std::make_shared<Model>("solarSystem/pluto/pluto.obj");
        break;
    case ModelType::Sun:
        modelPtr = std::make_shared<Model>("solarSystem/sun/sol.obj");
        break;
    case ModelType::Jupiter:
        modelPtr = std::make_shared<Model>("solarSystem/jupiter/jupiter.obj");
        break;
    case ModelType::Neptune:
        modelPtr = std::make_shared<Model>("solarSystem/neptune/neptuno.obj");
        break;
    case ModelType::Saturn:
        modelPtr = std::make_shared<Model>("solarSystem/saturn/saturno.obj");
        break;
    case ModelType::Login:
        modelPtr = std::make_shared<Model>("login.obj");
        break;
    default:
        std::cerr << "[ModelManager] createModel: unknown ModelType\n";
        return nullptr;
    }

    if (!modelPtr) return nullptr;
    // Store created model in cache and return shared_ptr to it.
    cache[type] = std::move(modelPtr);
    std::cerr << "[ModelManager] created model for type " << static_cast<int>(type) << "\n";
    return cache[type];
}