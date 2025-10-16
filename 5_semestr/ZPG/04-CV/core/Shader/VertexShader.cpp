#include "VertexShader.h"
#include <iostream>

VertexShader::VertexShader() : Shader() {
    shaderID = glCreateShader(GL_VERTEX_SHADER);
}

bool VertexShader::compile(const std::string& source) {
    if (shaderID == 0) {
        std::cerr << "Vertex shader not created!" << std::endl;
        return false;
    }
    
    const char* sourcePtr = source.c_str();
    glShaderSource(shaderID, 1, &sourcePtr, NULL);
    glCompileShader(shaderID);
    
    // Kontrola kompilace
    GLint status;
    glGetShaderiv(shaderID, GL_COMPILE_STATUS, &status);
    if (status != GL_TRUE) {
        GLint infoLogLength;
        glGetShaderiv(shaderID, GL_INFO_LOG_LENGTH, &infoLogLength);
        if (infoLogLength > 0) {
            GLchar* strInfoLog = new GLchar[infoLogLength + 1];
            glGetShaderInfoLog(shaderID, infoLogLength, NULL, strInfoLog);
            std::cerr << "Vertex shader compile error: " << strInfoLog << std::endl;
            delete[] strInfoLog;
        }
        return false;
    }
    return true;
}