#pragma once
#include <GL/glew.h>
#include <string>

class Shader {
protected:
    GLuint shaderID;
    
    // ShaderProgram potřebuje přístup k ID
    friend class ShaderProgram;
    
public:
    Shader();
    virtual ~Shader();
    
    virtual bool compile(const std::string& source) = 0;
    bool isCompiled() const;
};