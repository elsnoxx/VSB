#pragma once
#include <GL/glew.h>
#include <string>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <sstream>

class ShaderLoader {
public:
    static std::string loadFile(const char* fname);
    
    // Factory metody pro vytváření shaderů ze souborů
    static class VertexShader* createVertexShader(const char* filename);
    static class FragmentShader* createFragmentShader(const char* filename);
    
    // Factory metoda pro vytvoření kompletního ShaderProgram ze souborů
    static class ShaderProgram* createShaderProgram(const char* vertexFile, const char* fragmentFile);
};