#pragma once
#include "Shader.h"

class VertexShader : public Shader {
public:
    bool compile(const std::string& source) {
        return Shader::compile(source, GL_VERTEX_SHADER);
    }
};