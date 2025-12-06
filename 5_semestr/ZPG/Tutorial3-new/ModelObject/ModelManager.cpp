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

namespace {
    inline int countVertices(size_t bytes) {
        return static_cast<int>(bytes / (6 * sizeof(float)));
    }
}

ModelManager& ModelManager::instance() {
    static ModelManager inst;
    return inst;
}

Model* ModelManager::get(ModelType type) {
    auto it = cache.find(type);
    if (it != cache.end()) {
        return it->second.get();
    }

    // lazy create
    return createModel(type);
}

Model* ModelManager::createModel(ModelType type) {
    std::unique_ptr<Model> modelPtr;

    switch (type) {
    case ModelType::Sphere:
        modelPtr = std::unique_ptr<Model>(new Model(sphere, sizeof(sphere), countVertices(sizeof(sphere))));
        break;
    case ModelType::Tree:
        modelPtr = std::unique_ptr<Model>(new Model(tree, sizeof(tree), countVertices(sizeof(tree))));
        break;
    case ModelType::Bushes:
        modelPtr = std::unique_ptr<Model>(new Model(bushes, sizeof(bushes), countVertices(sizeof(bushes))));
        break;
    case ModelType::SuziFlat:
        modelPtr = std::unique_ptr<Model>(new Model(suziFlat, sizeof(suziFlat), countVertices(sizeof(suziFlat))));
        break;
    case ModelType::SuziSmooth:
        modelPtr = std::unique_ptr<Model>(new Model(suziSmooth, sizeof(suziSmooth), countVertices(sizeof(suziSmooth))));
        break;
    case ModelType::Plain:
        modelPtr = std::unique_ptr<Model>(new Model(plain, sizeof(plain), countVertices(sizeof(plain))));
        break;
    case ModelType::Gift:
        modelPtr = std::unique_ptr<Model>(new Model(gift, sizeof(gift), countVertices(sizeof(gift))));
        break;
    case ModelType::House:
        modelPtr = std::unique_ptr<Model>(new Model("house.obj"));
        break;
    case ModelType::Formula1:
        modelPtr = std::unique_ptr<Model>(new Model("formula1.obj"));
        break;
    case ModelType::Cube:
        modelPtr = std::unique_ptr<Model>(new Model("cube.obj"));
        break;
    case ModelType::Square:
        modelPtr = std::unique_ptr<Model>(new Model("square.obj"));
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