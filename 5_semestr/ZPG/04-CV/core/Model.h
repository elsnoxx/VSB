#pragma once
#pragma once
#include <GL/glew.h>

class Model {
private:
    GLuint VAO, VBO;
    int vertexCount;
    
public:
    Model(const float* vertices, int size, int vertCount);
    ~Model();
    
    void bind() const;
    void draw() const;
    int getVertexCount() const { return vertexCount; }
};