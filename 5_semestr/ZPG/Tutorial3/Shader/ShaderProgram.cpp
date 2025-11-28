#include "ShaderProgram.h"

ShaderProgram::ShaderProgram(const VertexShader& vertex, const FragmentShader& fragment) {
    id = glCreateProgram();
    glAttachShader(id, vertex.getId());
    glAttachShader(id, fragment.getId());
    glLinkProgram(id);

    GLint success;
    glGetProgramiv(id, GL_LINK_STATUS, &success);
    if (!success) {
        char log[512];
        glGetProgramInfoLog(id, 512, nullptr, log);
        std::cerr << "Shader program linking error: " << log << std::endl;
    }
}

ShaderProgram::~ShaderProgram() {
    glDeleteProgram(id);
}

void ShaderProgram::use() const {
    glUseProgram(id);
}

void ShaderProgram::setUniformMat4(const char* name, const glm::mat4& matrix) const {
    GLint location = glGetUniformLocation(id, name);
    if (location != -1) glUniformMatrix4fv(location, 1, GL_FALSE, glm::value_ptr(matrix));
}

void ShaderProgram::setUniformVec3(const char* name, const glm::vec3& vec) const {
    GLint location = glGetUniformLocation(id, name);
    if (location != -1) glUniform3fv(location, 1, glm::value_ptr(vec));
}

void ShaderProgram::setUniformFloat(const char* name, float value) const {
    GLint location = glGetUniformLocation(id, name);
    if (location != -1) glUniform1f(location, value);
}

void ShaderProgram::setUniformInt(const char* name, int value) const {
    GLint location = glGetUniformLocation(id, name);
    if (location != -1) glUniform1i(location, value);
}
