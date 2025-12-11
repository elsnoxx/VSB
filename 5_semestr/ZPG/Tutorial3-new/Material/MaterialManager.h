#pragma once
#include "MaterialType.h"
#include "MaterialData.h"

class MaterialManager {
public:
    static MaterialManager& instance() {
        static MaterialManager inst;
        return inst;
    }

    // Return pointer to MaterialData (non-owning). Implementation may call an
    // existing free function `getMaterial()` from Materials.cpp or use the
    // internal predefined materials provided by MaterialManager.
    const MaterialData* get(MaterialType mt) const;

private:
    MaterialManager() = default;
    ~MaterialManager() = default;
    MaterialManager(const MaterialManager&) = delete;
    MaterialManager& operator=(const MaterialManager&) = delete;
};