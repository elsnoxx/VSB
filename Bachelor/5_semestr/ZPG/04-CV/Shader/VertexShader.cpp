#include "VertexShader.h"

VertexShader::VertexShader(const std::string& source)
    : Shader(GL_VERTEX_SHADER, source) {
}
