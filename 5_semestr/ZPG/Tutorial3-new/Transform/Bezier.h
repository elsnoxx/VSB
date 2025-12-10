// ...existing code...
#pragma once
#include "AbstractTransform.h"
#include <vector>
#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>

// jednoduchý Bezier transform: pás po kubických segmentech (4 control points na segment)
class Bezier : public AbstractTransform {
public:
    // controlPoints: musí obsahovat 4 + 3*k prvků (tj. sousední segmenty sdílejí body)
    // durationSeconds = how many seconds to travel the entire spline (all segments)
    // orient = pokud true, transformuje také orientaci podle tangenty (forward = směrový vektor)
    // upVec = referenční up v world space (často {0,1,0})
    // pivotOffset = lokální posun modelu (korekce pivotu)
    Bezier(const std::vector<glm::vec3>& controlPoints,
           float durationSeconds = 6.0f,
           bool loop = true,
           bool orient = false,
           const glm::vec3& upVec = glm::vec3(0.0f, 1.0f, 0.0f),
           const glm::vec3& pivotOffset = glm::vec3(0.0f));

    // vrací matici transformace pro aktuální čas
    glm::mat4 getMatrix() const override;

    // helper: vyhodnotí jeden kubický segment (p0..p3) pro lokální t in [0,1]
    static glm::vec3 evalCubic(const glm::vec3& p0,
                               const glm::vec3& p1,
                               const glm::vec3& p2,
                               const glm::vec3& p3,
                               float t);

private:
    // numerická derivace (malý epsilon)
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