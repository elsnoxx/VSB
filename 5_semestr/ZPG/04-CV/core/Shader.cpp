#include "Shader.h"
#include <iostream>

Shader::~Shader() {
    if (id) glDeleteShader(id);
}

bool Shader::compile(const std::string& source, GLenum type) {
    id = glCreateShader(type);
    const char* src = source.c_str();
    glShaderSource(id, 1, &src, nullptr);
    glCompileShader(id);

    GLint status;
    glGetShaderiv(id, GL_COMPILE_STATUS, &status);
    if (status != GL_TRUE) {
        char buffer[512];
        glGetShaderInfoLog(id, 512, nullptr, buffer);
        std::cerr << "Shader compile error: " << buffer << std::endl;
        return false;
    }
    return true;
}