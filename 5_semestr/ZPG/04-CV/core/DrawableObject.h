#pragma once
#include "Model.h"
#include "Shader/ShaderProgram.h"
#include "Transformation/Transformation.h"
#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <memory>

class DrawableObject {
private:
    Model* model;
    ShaderProgram* shaderProgram;
    std::shared_ptr<Transformation> transformation;
    
public:
    DrawableObject(Model* model, ShaderProgram* shaderProgram);
    ~DrawableObject() = default;
    
    void draw() const;
    
    // Transformace
    void setTransformation(std::shared_ptr<Transformation> transformation);
    void addTransformation(std::shared_ptr<TransformationAbstract> transform);
    std::shared_ptr<Transformation> getTransformation() const { return transformation; }
    glm::mat4 getModelMatrix() const;
};