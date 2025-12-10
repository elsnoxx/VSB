#pragma once
#include "AbstractTransform.h"
#include <vector>
#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>

// jednoduchý Bezier transform: pás po kubických segmentech (4 control points na segment)
class Bezier : public AbstractTransform {
public:
    // controlPoints: musí obsahovat 4 + 3*k prvkù (tj. sousední segmenty sdílejí body)
    Bezier(const std::vector<glm::vec3>& controlPoints, float speed = 1.0f, bool loop = true);

    // vrací matici transformace pro aktuální èas
    glm::mat4 getMatrix() const override;

    // helper: vyhodnotí jeden kubický segment (p0..p3) pro lokální t in [0,1]
    static glm::vec3 evalCubic(const glm::vec3& p0, const glm::vec3& p1, const glm::vec3& p2, const glm::vec3& p3, float t);

private:
    std::vector<glm::vec3> pts;
    float speed;
    bool loop;
    double startTime;
};