#pragma once
#include <GL/glew.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include "VertexShader.h"
#include "FragmentShader.h"
#include "../Camera/Camera.h"
#include "../Observer/Observer.h"


class ShaderProgram : public Observer {
protected:
    GLuint id;
	Camera* camera = nullptr;

public:
    ShaderProgram(const char* vertexSrc, const char* fragmentSrc);
    ~ShaderProgram();

	//observer method
    void update(ObservableSubjects subject) override;

    void use() const;

    void attachCamera(Camera* cam);

    void setUniform(const char* name, const glm::mat4& matrix) const;
    void setUniform(const char* name, const glm::vec3& vec) const;
    void setUniform(const char* name, float value) const;
    void setUniform(const char* name, int value) const;
};
