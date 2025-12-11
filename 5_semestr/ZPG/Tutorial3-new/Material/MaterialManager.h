#pragma once
#include "MaterialType.h"
#include "MaterialData.h"

class MaterialManager {
public:
    static MaterialManager& instance() {
        static MaterialManager inst;
        return inst;
    }

    // Vrací ukazatel na MaterialData (ne-vlastnìný). Implementace mùže volat stávající getMaterial() z Materials.cpp
    const MaterialData* get(MaterialType mt) const;

private:
    MaterialManager() = default;
    ~MaterialManager() = default;
    MaterialManager(const MaterialManager&) = delete;
    MaterialManager& operator=(const MaterialManager&) = delete;
};