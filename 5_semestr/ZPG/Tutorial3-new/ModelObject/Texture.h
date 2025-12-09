#pragma once
#include <GL/glew.h>
#include <string>

class Texture {
public:
    Texture() = default;
    // load image from disk (path relative to project)
    Texture(const std::string& path);
    // create wrapper from existing GL texture id
    Texture(GLuint existingId);
    ~Texture();

    void bind(unsigned int unit = 0) const;
    GLuint getId() const { return id; }
    // alias pro kompatibilitu s kódem, který volal getID()
    GLuint getID() const { return id; }
    bool valid() const { return id != 0; }

private:
    GLuint id = 0;
    std::string path;
};