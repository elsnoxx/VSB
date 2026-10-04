#pragma once
#include <GL/glew.h>



#include "Model.h"
#include <vector>
#include <string>
#include <iostream>
#include <stdexcept>

class Model {
private:
    GLuint VAO;
    GLuint VBO;
    int vertexCount;

public:
    Model(const float* data, size_t size, int count);
    Model(const char* name);
    ~Model();

    void draw() const;
};
