#pragma once
#include <GL/glew.h>

class Texture {
public:
    explicit Texture(GLuint id = 0) : id(id) {}
    ~Texture() { if (id) glDeleteTextures(1, &id); }
    GLuint getID() const { return id; }
private:
    GLuint id;
};