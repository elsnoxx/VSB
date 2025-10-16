#pragma once
#include <glm/mat4x4.hpp>
#include <vector>
#include <memory>

class TransformationAbstract {
public:
    virtual ~TransformationAbstract() = default;
    virtual glm::mat4 getMatrix() const = 0;
    
    // Composite pattern methods
    virtual void add(std::shared_ptr<TransformationAbstract> transformation) {}
    virtual void remove(std::shared_ptr<TransformationAbstract> transformation) {}
    virtual bool isComposite() const { return false; }
};