#pragma once
#include <vector>
#include "../DrawableObject.h"

class Scene {
public:
    Scene() = default;

    void addObject(DrawableObject* obj);

    void draw();

private:
    std::vector<DrawableObject*> objects;
};
