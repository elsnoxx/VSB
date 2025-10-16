#pragma once
#include "Shader.h"

class VertexShader : public Shader {
public:
    VertexShader();
    virtual ~VertexShader() = default;
    
    bool compile(const std::string& source) override;
};