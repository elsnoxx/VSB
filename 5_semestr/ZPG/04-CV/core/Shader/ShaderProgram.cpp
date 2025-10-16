#include "ShaderProgram.h"
#include "VertexShader.h"
#include "FragmentShader.h"
#include <iostream>
#include <glm/gtc/type_ptr.hpp>

ShaderProgram::ShaderProgram(VertexShader* vs, FragmentShader* fs) 
    : vertexShader(vs), fragmentShader(fs), programID(0) {
    
    if (!vs || !fs || !vs->isCompiled() || !fs->isCompiled()) {
        std::cerr << "Invalid or uncompiled shaders provided to ShaderProgram" << std::endl;
        return;
    }
    
    programID = glCreateProgram();
    
    // Používáme friend přístup k shaderID
    glAttachShader(programID, vs->shaderID);
    glAttachShader(programID, fs->shaderID);
    
    glLinkProgram(programID);
    
    // Kontrola linkování
    GLint status;
    glGetProgramiv(programID, GL_LINK_STATUS, &status);
    if (status != GL_TRUE) {
        GLint infoLogLength;
        glGetProgramiv(programID, GL_INFO_LOG_LENGTH, &infoLogLength);
        if (infoLogLength > 0) {
            GLchar* strInfoLog = new GLchar[infoLogLength + 1];
            glGetProgramInfoLog(programID, infoLogLength, NULL, strInfoLog);
            std::cerr << "Shader program link error: " << strInfoLog << std::endl;
            delete[] strInfoLog;
        }
    }
    
    // Detach shaders po linkování
    glDetachShader(programID, vs->shaderID);
    glDetachShader(programID, fs->shaderID);
}

ShaderProgram::~ShaderProgram() {
    if (programID) {
        glDeleteProgram(programID);
    }
    delete vertexShader;
    delete fragmentShader;
}

void ShaderProgram::use() const {
    if (programID && isLinked()) {
        glUseProgram(programID);
    }
}

bool ShaderProgram::isLinked() const {
    if (programID == 0) return false;
    
    GLint status;
    glGetProgramiv(programID, GL_LINK_STATUS, &status);
    return status == GL_TRUE;
}

void ShaderProgram::setUniform(const std::string& name, const glm::mat4& matrix) const {
    GLint location = glGetUniformLocation(programID, name.c_str());
    if (location != -1) {
        glUniformMatrix4fv(location, 1, GL_FALSE, glm::value_ptr(matrix));
    }
}

void ShaderProgram::setUniform(const std::string& name, float value) const {
    GLint location = glGetUniformLocation(programID, name.c_str());
    if (location != -1) {
        glUniform1f(location, value);
    }
}

void ShaderProgram::setUniform(const std::string& name, int value) const {
    GLint location = glGetUniformLocation(programID, name.c_str());
    if (location != -1) {
        glUniform1i(location, value);
    }
}

void ShaderProgram::setUniform(const std::string& name, const glm::vec3& vector) const {
    GLint location = glGetUniformLocation(programID, name.c_str());
    if (location != -1) {
        glUniform3fv(location, 1, glm::value_ptr(vector));
    }
}