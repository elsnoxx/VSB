#pragma once
#include "Shader.h"

class FragmentShader : public Shader {
public:
    FragmentShader();
    virtual ~FragmentShader() = default;
    
    bool compile(const std::string& source) override;
};