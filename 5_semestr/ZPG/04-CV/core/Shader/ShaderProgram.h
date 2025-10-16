#pragma once
#include <string>
#include <GL/glew.h>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

// Forward declarations
class VertexShader;
class FragmentShader;

class ShaderProgram {
private:
    GLuint programID;
    VertexShader* vertexShader;
    FragmentShader* fragmentShader;
    
public:
    ShaderProgram(VertexShader* vs, FragmentShader* fs);
    ~ShaderProgram();
    
    void use() const;
    bool isLinked() const;
    
    // Utility metody pro uniformy
    void setUniform(const std::string& name, const glm::mat4& matrix) const;
    void setUniform(const std::string& name, float value) const;
    void setUniform(const std::string& name, int value) const;
    void setUniform(const std::string& name, const glm::vec3& vector) const;
};