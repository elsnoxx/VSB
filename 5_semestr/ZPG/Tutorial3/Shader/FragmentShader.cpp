#include "FragmentShader.h"

FragmentShader::FragmentShader(const std::string& source)
    : Shader(GL_FRAGMENT_SHADER, source) {
}