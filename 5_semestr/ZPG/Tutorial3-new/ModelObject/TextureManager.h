#pragma once
#include <memory>
#include <unordered_map>
#include <string>
#include "Texture.h"
#include "TextureType.h"

class TextureManager {
public:
    static TextureManager& instance();

    // get cached texture by enum (loads on first request)
    std::shared_ptr<Texture> get(TextureType t);

    // load from arbitrary path (cesta relativní k projektu)
    std::shared_ptr<Texture> loadFromFile(const std::string& path);

    // convenience alias
    std::shared_ptr<Texture> get(const std::string& path) { return loadFromFile(path); }

private:
    TextureManager() = default;
    ~TextureManager() = default;
    TextureManager(const TextureManager&) = delete;
    TextureManager& operator=(const TextureManager&) = delete;

    // interní cache
    std::unordered_map<TextureType, std::shared_ptr<Texture>> cacheByType;
    std::unordered_map<std::string, std::shared_ptr<Texture>> cacheByPath;

    // pomocná metoda - naète GL texturu a vrátí id
    GLuint LoadTexture(const std::string& path);
};