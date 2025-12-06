#pragma once
#include <GL/glew.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include "VertexShader.h"
#include "FragmentShader.h"
#include "../Observer/Observer.h"
#include "../Camera/Camera.h"



class ShaderProgram : public Observer {
private:
    GLuint id;

public:
    ShaderProgram(const VertexShader& vertex, const FragmentShader& fragment);
    ~ShaderProgram();

    void use() const;

    void updateCameraMatrices(const Camera& cam, float aspect);

    // Observer callback:
    void onCameraChanged(const Camera& cam) override;
    void updateMatricesInGPU() const;

    void setUniformMat4(const char* name, const glm::mat4& matrix) const;
    void setUniformVec3(const char* name, const glm::vec3& vec) const;
    void setUniformFloat(const char* name, float value) const;
    void setUniformInt(const char* name, int value) const;
};
