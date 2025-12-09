#include "Texture.h"
#include "../lib/std_image/stb-image-header.h"
#include <iostream>

Texture::Texture(const std::string& p) : path(p) {
    int w, h, ch;
    stbi_set_flip_vertically_on_load(1);
    unsigned char* data = stbi_load(path.c_str(), &w, &h, &ch, 0);
    if (!data) {
        std::cerr << "[Texture] stbi_load failed: " << path << "\n";
        id = 0;
        return;
    }
    GLenum format = (ch == 4) ? GL_RGBA : GL_RGB;
    glGenTextures(1, &id);
    glBindTexture(GL_TEXTURE_2D, id);
    glTexImage2D(GL_TEXTURE_2D, 0, format, w, h, 0, format, GL_UNSIGNED_BYTE, data);
    glGenerateMipmap(GL_TEXTURE_2D);
    // basic params
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
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