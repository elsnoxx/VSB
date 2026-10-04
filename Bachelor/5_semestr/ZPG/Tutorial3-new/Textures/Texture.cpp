#include "Texture.h"
#include "../lib/std_image/stb-image-header.h"
#include <iostream>

Texture::Texture(const std::string& p) : path(p) {
    int w, h, ch;
    // stb_image: flip the image vertically on load so texture coordinates
    // (0,0) is bottom-left which matches OpenGL convention in many shaders.
    stbi_set_flip_vertically_on_load(1);

    // Load image file. `ch` will contain number of channels (3=RGB,4=RGBA).
    // We pass 0 for desired_channels to keep the file's native channel count.
    unsigned char* data = stbi_load(path.c_str(), &w, &h, &ch, 0);
    if (!data) {
        std::cerr << "[Texture] stbi_load failed: " << path << "\n";
        id = 0;
        return;
    }
    // Determine GL format based on number of channels.
    // If the file has an alpha channel (4), use GL_RGBA, otherwise GL_RGB.
    GLenum format = (ch == 4) ? GL_RGBA : GL_RGB;
    glGenTextures(1, &id);
    glBindTexture(GL_TEXTURE_2D, id);
    glTexImage2D(GL_TEXTURE_2D, 0, format, w, h, 0, format, GL_UNSIGNED_BYTE, data);
    glGenerateMipmap(GL_TEXTURE_2D);
    // basic params
    // Set texture filtering: trilinear minification (mipmaps) and linear magnification.
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // Note: consider setting wrap modes (GL_TEXTURE_WRAP_S/T) if you need repeat/clamp behavior.
    glBindTexture(GL_TEXTURE_2D, 0);
    stbi_image_free(data);
}

// new ctor from existing GL id (used by TextureManager)
Texture::Texture(GLuint existingId) {
    id = existingId;
}

Texture::~Texture() {
    if (id) glDeleteTextures(1, &id);
}

void Texture::bind(unsigned int unit) const {
    glActiveTexture(GL_TEXTURE0 + unit);
    glBindTexture(GL_TEXTURE_2D, id);
}