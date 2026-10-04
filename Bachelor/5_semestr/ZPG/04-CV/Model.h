#pragma once
#include <GL/glew.h>

class Model {
private:
    GLuint VAO;
    GLuint VBO;
    int vertexCount;

public:
    Model(const float* data, size_t size, int count);
    ~Model();

    void draw() const;
};
