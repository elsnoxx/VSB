#include "TextureManager.h"
#define STB_IMAGE_IMPLEMENTATION
#include "../lib/std_image/stb-image-header.h"
#include <iostream>

// instance() u� bylo OK
TextureManager& TextureManager::instance() {
    static TextureManager inst;
    return inst;
}

// implementace get(TextureType) -- u� v souboru m�l k�d, ��dn� zm�ny pot�eba,
// ale ujist�me se, �e signatury odpov�daj� hlavi�ce:
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
    case TextureType::Grass:
        path = "ModelObject/textures/grass.png";
        break;
    case TextureType::Venus:
        path = "ModelObject/assets/solarSystem/venus/Textures/Atmosphere_2K.png";
        break;
    case TextureType::Moon:
        path = "ModelObject/assets/solarSystem/moon/Textures/Bump_2K.png";
        break;
    case TextureType::Mercury:
        path = "ModelObject/assets/solarSystem/mercury/Textures/Bump_1K.png";
        break;
    case TextureType::Earth:
        path = "ModelObject/assets/solarSystem/earth/Textures/Diffuse_2K.png";
        break;
    case TextureType::Mars:
        path = "ModelObject/assets/solarSystem/mars/Textures/Diffuse_2K.png";
        break;
    case TextureType::Uranus:
        path = "ModelObject/assets/solarSystem/urano/13907_Uranus_planet_diff.jpg";
        break;
    case TextureType::Pluto:
        path = "ModelObject/assets/solarSystem/pluto/pluto.PNG";
        break;
    case TextureType::Sun:
        path = "ModelObject/assets/solarSystem/sun/2k_sun.jpg";
        break;
    case TextureType::Jupiter:
        path = "ModelObject/assets/solarSystem/jupiter/textures/descarga.jpeg";
        break;
    case TextureType::Neptune:
        path = "ModelObject/assets/solarSystem/neptune/2k_neptune.jpg";
        break;
    case TextureType::Saturn:
        path = "ModelObject/assets/solarSystem/saturn/2k_saturn.jpg";
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