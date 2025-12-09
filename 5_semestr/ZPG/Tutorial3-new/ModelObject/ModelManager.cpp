#include "ModelManager.h"
#include <iostream>

// include model data (cesty relativni k ModelManager.cpp v ModelObject/)
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
    return createModel(type);
}

std::shared_ptr<Model> ModelManager::createModel(ModelType type) {
    std::shared_ptr<Model> modelPtr;

    switch (type) {
    case ModelType::Triangle:
        modelPtr = std::make_shared<Model>(tri, sizeof(sphere), countVertices(sizeof(sphere)));
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
    default:
        std::cerr << "[ModelManager] createModel: unknown ModelType\n";
        return nullptr;
    }

    if (!modelPtr) return nullptr;
    Model* ptr = modelPtr.get();
    cache[type] = std::move(modelPtr);
    std::cerr << "[ModelManager] created model for type " << static_cast<int>(type) << "\n";
    return ptr;
}