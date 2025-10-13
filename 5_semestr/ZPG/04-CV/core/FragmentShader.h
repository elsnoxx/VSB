#pragma once
#include "Shader.h"

class FragmentShader : public Shader {
public:
    bool compile(const std::string& source) {
        return Shader::compile(source, GL_FRAGMENT_SHADER);
    }
};