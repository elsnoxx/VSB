#pragma once
#include <GL/glew.h>
#include <vector>

class Model {
public:
    Model(const float* data, size_t count, GLuint stride = 6);
    ~Model();

    void bind() const;
    void unbind() const;
    GLuint getVAO() const { return VAO; }
    GLuint getVBO() const { return VBO; }
    size_t getVertexCount() const { return vertexCount; }

private:
    GLuint VAO, VBO;
    size_t vertexCount;
};