#pragma once
#include "AbstractTransform.h"
#include <vector>
#include <memory>

using namespace std;

class Transform {
public:
	Transform() = default;

    void addTransform(shared_ptr<AbstractTransform> t);

    glm::mat4 getMatrix() const;

private:
    vector<shared_ptr<AbstractTransform>> transforms;
};
