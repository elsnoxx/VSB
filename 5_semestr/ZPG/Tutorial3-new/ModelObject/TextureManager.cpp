#include "TextureManager.h"
#define STB_IMAGE_IMPLEMENTATION
#include "../lib/std_image/stb-image-header.h"
#include <iostream>

TextureManager& TextureManager::instance() {
    static TextureManager inst;
    return inst;
}

std::shared_ptr<Texture> TextureManager::get(TextureType t) {
    auto it = cacheByType.find(t);
    if (it != cacheByType.end()) return it->second;

    std::string path;
    switch (t) {
    case TextureType::Fiona: 
        path = "ModelObject/textures/fiona.png"; 
        break;
    case TextureType::Shrek: 
        path = "ModelObject/textures/shrek.png"; 
        break;
    case TextureType::Toilet: 
        path = "ModelObject/textures/toiled.jpg"; 
        break;
    case TextureType::Teren: 
        path = "ModelObject/textures/grass.png"; 
        break;
    case TextureType::WoodenFence:
        path = "ModelObject/textures/wooden_fence.png";
        break;
    default: 
        path = ""; 
        break;
    }

    if (path.empty()) return nullptr;
    GLuint id = LoadTexture(path);
    if (!id) return nullptr;
    auto tex = std::make_shared<Texture>(id);
    cacheByType[t] = tex;
    return tex;
}

std::shared_ptr<Texture> TextureManager::loadFromFile(const std::string& path) {
    auto it = cacheByPath.find(path);
    if (it != cacheByPath.end()) return it->second;
    GLuint id = LoadTexture(path);
    if (!id) return nullptr;
    auto tex = std::make_shared<Texture>(id);
    cacheByPath[path] = tex;
    return tex;
}

GLuint TextureManager::LoadTexture(const std::string& path) {
    int w, h, chans;
    stbi_set_flip_vertically_on_load(true);
    unsigned char* data = stbi_load(path.c_str(), &w, &h, &chans, 4);
    if (!data) {
        std::cerr << "Failed to load texture: " << path << " - " << stbi_failure_reason() << "\n";
        return 0;
    }

    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0,
        GL_RGBA, GL_UNSIGNED_BYTE, data);

    glGenerateMipmap(GL_TEXTURE_2D);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    stbi_image_free(data);
    return tex;
}
