#include "TextureManager.h"
#define STB_IMAGE_IMPLEMENTATION
#include "../lib/std_image/stb-image-header.h"
#include <iostream>
#include <algorithm> 

// TextureManager responsibilities:
// - Create and cache OpenGL textures from files or procedurally (solid colors).
// - Provide a single shared instance via `instance()`.
// - Keep two caches: `cacheByPath` (path->Texture) and `cacheByType` (TextureType->Texture).
//
// Notes on behavior:
// - Loaded textures are uploaded as RGBA (4 channels) to the GPU and mipmaps are generated.
// - The manager returns `std::shared_ptr<Texture>` so ownership is shared with callers.

GLuint TextureManager::CreateColorTexture(unsigned char r, unsigned char g, unsigned char b, unsigned char a) {
    unsigned char pixel[4] = { r, g, b, a };
    GLuint tex = 0;
    glGenTextures(1, &tex);
    if (tex == 0) return 0;
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixel);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);
    return tex;
}

std::shared_ptr<Texture> TextureManager::getColored(TextureType t, float r, float g, float b) {
    auto it = cacheByType.find(t);
    if (it != cacheByType.end()) return it->second;

    // clamp and convert to 0-255
    auto clamp01 = [](float v)->unsigned char {
        float c = v;
        if (c < 0.0f) c = 0.0f;
        if (c > 1.0f) c = 1.0f;
        return static_cast<unsigned char>(c * 255.0f);
    };
    unsigned char rc = clamp01(r);
    unsigned char gc = clamp01(g);
    unsigned char bc = clamp01(b);

    GLuint id = CreateColorTexture(rc, gc, bc, 255);
    if (!id) return nullptr;
    auto tex = std::make_shared<Texture>(id);
    cacheByType[t] = tex;
    return tex;
}

// Singleton accessor for the TextureManager
// Returns a reference to a single shared manager instance used throughout the app.
/**
 * instance
 * --------
 * Return the singleton TextureManager instance. The manager holds caches
 * and is intended to be used globally to avoid duplicate GPU textures.
 */
TextureManager& TextureManager::instance() {
    static TextureManager inst;
    return inst;
}

// Get texture by `TextureType` enum. If a texture is not cached yet, this
// function determines the file path (or creates a solid-color texture) and
// loads it, caching the resulting `Texture` object for future requests.
/**
 * get
 * ---
 * Retrieve a texture identified by the `TextureType` enum. This method:
 * 1) Checks `cacheByType` for an existing texture.
 * 2) If missing, maps the enum to a file path (or creates a colored texture),
 *    loads the texture with `LoadTexture()` and caches the result.
 * Parameters:
 *  - t: the TextureType to retrieve
 * Returns:
 *  - shared_ptr to the cached or newly loaded Texture, or nullptr on failure.
 */
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
    case TextureType::Red:
        return getColored(t, 1.0f, 0.0f, 0.0f);
    case TextureType::Yellow:
        return getColored(t, 1.0f, 1.0f, 0.0f);
    case TextureType::Green:
        return getColored(t, 0.0f, 1.0f, 0.0f);
    case TextureType::Blue:
        return getColored(t, 0.0f, 0.0f, 1.0f);
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
    // Upload RGBA texture data to GPU and generate mipmaps.
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0,
        GL_RGBA, GL_UNSIGNED_BYTE, data);

    glGenerateMipmap(GL_TEXTURE_2D);

    // Set common sampling/wrap parameters: repeat wrap and trilinear filtering.
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // Free CPU-side image memory and return the GL texture id.
    stbi_image_free(data);
    return tex;
}
/**
 * LoadTexture
 * -----------
 * Load an image file from disk and upload it as an RGBA OpenGL texture.
 * Parameters:
 *  - path: filesystem path to the image file
 * Returns:
 *  - GL texture id (non-zero) on success, 0 on failure.
 * Behavior/details:
 *  - Uses stb_image to load the file; requests 4 output channels (RGBA).
 *  - The image is flipped vertically on load to match OpenGL texture coordinate
 *    conventions used in the project.
 *  - Generates mipmaps and sets default wrap/filter parameters (REPEAT, trilinear).
 */