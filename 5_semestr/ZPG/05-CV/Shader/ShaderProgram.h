#pragma once
#include <GL/glew.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <vector>
#include "VertexShader.h"
#include "FragmentShader.h"

class Camera;

class ShaderProgram {
private:
    GLuint id;
    std::vector<Camera*> m_cameras; // podporuje více kamer (pro různé scény)

public:
    ShaderProgram(const VertexShader& vertex, const FragmentShader& fragment);
    ~ShaderProgram();

    void use() const;

    void setUniformMat4(const char* name, const glm::mat4& matrix) const;
    void setUniformVec3(const char* name, const glm::vec3& vec) const;
    void setUniformFloat(const char* name, float value) const;
    void setUniformInt(const char* name, int value) const;

    // spojení s kamerou (observer) - podporuje více kamer
    void addCamera(Camera* cam);
    void removeCamera(Camera* cam);
    void removeAllCameras();
    
    // metoda, kterou Camera volá při změně (observer callback)
    void onCameraChanged(Camera* camera);
    
    // aktualizuje uniformy z aktuálně aktivní kamery (první v seznamu)
    void updateCameraUniforms();
};
