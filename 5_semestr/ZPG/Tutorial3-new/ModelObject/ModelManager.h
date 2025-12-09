#pragma once
#include <unordered_map>
#include <memory>
#include "Model.h"
#include "ModelType.h"

class ModelManager {
public:
    static ModelManager& instance();

    // lazy-loading:
    std::shared_ptr<Model> get(ModelType type);

    ModelManager(const ModelManager&) = delete;
    ModelManager& operator=(const ModelManager&) = delete;

private:
    ModelManager() = default;
    ~ModelManager() = default;


    std::shared_ptr<Model> createModel(ModelType type);

    std::unordered_map<ModelType, std::shared_ptr<Model>> cache;
};