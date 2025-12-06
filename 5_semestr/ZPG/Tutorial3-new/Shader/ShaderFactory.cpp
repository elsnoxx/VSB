#include "ShaderFactory.h"

std::unordered_map<ShaderType, ShaderProgram*> ShaderFactory::cache;

std::string LoadFile(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "[ShaderFactory] ERROR: Cannot open " << path << "\n";
        return "";
    }
    std::stringstream buffer;
    buffer << f.rdbuf();
    std::string content = buffer.str();

    // Remove UTF-8 BOM if present (0xEF,0xBB,0xBF)
    if (content.size() >= 3
        && static_cast<unsigned char>(content[0]) == 0xEF
        && static_cast<unsigned char>(content[1]) == 0xBB
        && static_cast<unsigned char>(content[2]) == 0xBF) {
        content.erase(0, 3);
		std::cout << "[ShaderFactory] Removed UTF-8 BOM from " << path << "\n";
    }

    if (content.empty()) {
        std::cerr << "[ShaderFactory] ERROR: File EMPTY: " << path << "\n";
    }
    else {
        std::cerr << "[ShaderFactory] Loaded '" << path << "' (" << content.size() << " bytes)\n";
    }
    return content;
}


ShaderProgram* ShaderFactory::Get(ShaderType type) {
    // Pokud už shader v cache existuje → vrátíme ho
    if (cache.count(type))
        return cache[type];

    // Pokud neexistuje → načteme
    ShaderProgram* shader = LoadShader(type);
    cache[type] = shader;
    return shader;
}

ShaderProgram* ShaderFactory::LoadShader(ShaderType type) {
    std::string vertexSrc;
    std::string fragmentSrc;

    switch (type) {
    case ShaderType::Basic:
        vertexSrc = LoadFile(Config::VertexShadersPath + "normals.vert.glsl");
        fragmentSrc = LoadFile(Config::FragmentShadersPath + "normals.frag.glsl");
        break;

    case ShaderType::Phong:
        vertexSrc = LoadFile(Config::VertexShadersPath + "phong.vert.glsl");
        fragmentSrc = LoadFile(Config::FragmentShadersPath + "phong.frag.glsl");
        break;

    case ShaderType::Lambert:
        vertexSrc = LoadFile(Config::VertexShadersPath + "lambert.vert.glsl");
        fragmentSrc = LoadFile(Config::FragmentShadersPath + "lambert.frag.glsl");
        break;

    case ShaderType::Textured:
        vertexSrc = LoadFile(Config::VertexShadersPath + "textured.vert.glsl");
        fragmentSrc = LoadFile(Config::FragmentShadersPath + "textured.frag.glsl");
        break;
    }

    return new ShaderProgram(vertexSrc.c_str(), fragmentSrc.c_str());
}
