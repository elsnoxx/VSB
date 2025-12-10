#pragma once
#include "Transform.h"
#include <memory>
#include <vector>

class TransformNode : public std::enable_shared_from_this<TransformNode> {
public:
    using Ptr = std::shared_ptr<TransformNode>;

    TransformNode() = default;

    // add local transform (deleguje na Transform API)
    void addTransform(const std::shared_ptr<AbstractTransform>& tr) {
        local.addTransform(tr);
    }

    // add child and set parent
    void addChild(const Ptr& child) {
        if (!child) return;
        child->parent = shared_from_this();
        children.push_back(child);
    }

    // compute local matrix (compose local transforms)
    glm::mat4 computeLocalMatrix() const {
        return local.getMatrix(); // use your Transform::getMatrix()
    }

    // recursive world matrix: parent_world * local
    glm::mat4 computeWorldMatrix() const {
        glm::mat4 localM = computeLocalMatrix();
        if (auto p = parent.lock()) {
            return p->computeWorldMatrix() * localM;
        }
        return localM;
    }

    const std::vector<Ptr>& getChildren() const { return children; }

private:
    Transform local;
    std::vector<Ptr> children;
    std::weak_ptr<TransformNode> parent;
};