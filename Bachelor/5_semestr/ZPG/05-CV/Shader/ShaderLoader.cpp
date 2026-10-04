#include "ShaderLoader.h"
#include "VertexShader.h"
#include "FragmentShader.h"
#include "ShaderProgram.h"

std::string ShaderLoader::loadFile(const char* fname) {
    std::ifstream file;
    std::stringstream buf;
    std::string ret = "";
    
    file.open(fname, std::ios::in);
    if (file.is_open()) {
        buf << file.rdbuf();
        ret = buf.str();
    }
    else {
        std::cerr << "Could not open " << fname << " for reading!" << std::endl;
    }
    file.close();
    return ret;
}

VertexShader* ShaderLoader::createVertexShader(const char* filename) {
    std::string source = loadFile(filename);
    if (source.empty()) {
        std::cerr << "Failed to load vertex shader from: " << filename << std::endl;
        return nullptr;
    }
    return new VertexShader(source);
}

FragmentShader* ShaderLoader::createFragmentShader(const char* filename) {
    std::string source = loadFile(filename);
    if (source.empty()) {
        std::cerr << "Failed to load fragment shader from: " << filename << std::endl;
        return nullptr;
    }
    return new FragmentShader(source);
}

ShaderProgram* ShaderLoader::createShaderProgram(const char* vertexFile, const char* fragmentFile) {
    VertexShader* vertex = createVertexShader(vertexFile);
    FragmentShader* fragment = createFragmentShader(fragmentFile);
    
    if (!vertex || !fragment) {
        delete vertex;
        delete fragment;
        return nullptr;
    }
    
    ShaderProgram* program = new ShaderProgram(*vertex, *fragment);
    
    // Shadery můžeme smazat po vytvoření programu
    delete vertex;
    delete fragment;
    
    return program;
}