#pragma once
#include <GL/glew.h>
#include <string>
#include <iostream>

class Shader {
protected:
    GLuint id;

public:
    Shader(GLenum type, const std::string& source);

    GLuint getId() const;

    virtual ~Shader();
};
