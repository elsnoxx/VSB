#include "Shader.h"

Shader::Shader(GLenum type, const std::string& source) {
    id = glCreateShader(type);
    const char* src = source.c_str();
    glShaderSource(id, 1, &src, nullptr);
    glCompileShader(id);

    GLint success;
    glGetShaderiv(id, GL_COMPILE_STATUS, &success);
    if (!success) {
        char log[512];
        glGetShaderInfoLog(id, 512, nullptr, log);
        std::cerr << "Shader compilation error: " << log << std::endl;
    }
}

Shader::~Shader() {
    glDeleteShader(id);
}

GLuint Shader::getId() const{ 
	return id; 
}