#pragma once
#include <GL/glew.h>
#include <string>

class Shader {
protected:
    GLuint id;
public:
    Shader() : id(0) {}
    virtual ~Shader();
    GLuint getId() const { return id; }
    bool compile(const std::string& source, GLenum type);
};