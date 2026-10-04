#pragma once
#include "../Shader/ShaderProgram.h"
#include <unordered_map>
#include <fstream>
#include <sstream>
#include "ShaderType.h"
#include "../Config.h"

class ShaderFactory {
public:
    static ShaderProgram* Get(ShaderType type);

private:
    static ShaderProgram* LoadShader(ShaderType type);
    static std::unordered_map<ShaderType, ShaderProgram*> cache;
};
