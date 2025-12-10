// ...existing code...
#pragma once
#include "AbstractTransform.h"
#include <vector>
#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>

// simple Bezier transform: a strip of cubic segments (4 control points per segment)
class Bezier : public AbstractTransform {
public:
    // controlPoints: must contain 4 + 3*k elements (adjacent segments share points)
    // durationSeconds = how many seconds to travel the entire spline (all segments)
    // orient = if true, also orient the object according to the tangent (forward = direction vector)
    // upVec = reference up in world space (commonly {0,1,0})
    // pivotOffset = local offset for the model (pivot correction)
    Bezier(const std::vector<glm::vec3>& controlPoints,
           float durationSeconds = 6.0f,
           bool loop = true,
           bool orient = false,
           const glm::vec3& upVec = glm::vec3(0.0f, 1.0f, 0.0f),
           const glm::vec3& pivotOffset = glm::vec3(0.0f));

    // returns the transformation matrix for the current time
    glm::mat4 getMatrix() const override;

    // helper: evaluate one cubic segment (p0..p3) for local t in [0,1]
    static glm::vec3 evalCubic(const glm::vec3& p0,
                               const glm::vec3& p1,
                               const glm::vec3& p2,
                               const glm::vec3& p3,
                               float t);

private:
    // numerical derivative (small epsilon)
    static glm::vec3 evalDerivativeNumeric(const std::vector<glm::vec3>& pts, int base, float t);

    std::vector<glm::vec3> pts;
    float duration; // seconds for whole spline
    bool loop;
    bool orientToTangent;
    glm::vec3 up;
    glm::vec3 pivotOffset;
    double startTime;
};
// ...existing code...