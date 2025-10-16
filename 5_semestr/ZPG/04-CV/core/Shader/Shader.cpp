#include "Shader.h"
#include <iostream>

Shader::Shader() : shaderID(0) {
}

Shader::~Shader() {
    if (shaderID != 0) {
        glDeleteShader(shaderID);
    }
}

bool Shader::isCompiled() const {
    if (shaderID == 0) return false;
    
    GLint status;
    glGetShaderiv(shaderID, GL_COMPILE_STATUS, &status);
    return status == GL_TRUE;
}