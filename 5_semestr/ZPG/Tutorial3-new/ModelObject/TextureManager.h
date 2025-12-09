#pragma once
#include <memory>
#include <unordered_map>
#include <string> 
#include "Texture.h"
#include "TextureType.h"

class TextureManager {
public:
    static TextureManager& instance();

    std::shared_ptr<Texture> get(TextureType t);
    std::shared_ptr<Texture> loadFromFile(const std::string& path);

private:
    TextureManager() = default;

    GLuint LoadTexture(const std::string& path);

    std::unordered_map<TextureType, std::shared_ptr<Texture>> cacheByType;
    std::unordered_map<std::string, std::shared_ptr<Texture>> cacheByPath;
};