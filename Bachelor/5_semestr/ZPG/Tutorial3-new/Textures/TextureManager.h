#pragma once
#include <memory>
#include <unordered_map>
#include <string>
#include <GL/glew.h> 
#include "Texture.h"
#include "TextureType.h"

class TextureManager {
public:
    static TextureManager& instance();

    // get cached texture by enum (loads on first request)
    std::shared_ptr<Texture> get(TextureType t);

    // load from arbitrary path (cesta relativn� k projektu)
    std::shared_ptr<Texture> loadFromFile(const std::string& path);

    // convenience alias
    std::shared_ptr<Texture> get(const std::string& path) { return loadFromFile(path); }

    std::shared_ptr<Texture> getColored(TextureType t, float r, float g, float b);

private:
    TextureManager() = default;
    ~TextureManager() = default;
    TextureManager(const TextureManager&) = delete;
    TextureManager& operator=(const TextureManager&) = delete;

    // intern� cache
    std::unordered_map<TextureType, std::shared_ptr<Texture>> cacheByType;
    std::unordered_map<std::string, std::shared_ptr<Texture>> cacheByPath;

    // pomocn� metoda - na�te GL texturu a vr�t� id
    GLuint LoadTexture(const std::string& path);
    GLuint CreateColorTexture(unsigned char r, unsigned char g, unsigned char b, unsigned char a = 255);
};