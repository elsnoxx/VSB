#include "Bezier.h"
#include <GLFW/glfw3.h>
#include <glm/gtc/matrix_transform.hpp>

Bezier::Bezier(const std::vector<glm::vec3>& controlPoints, float speed_, bool loop_)
    : pts(controlPoints), speed(speed_), loop(loop_), startTime(glfwGetTime())
{
}

glm::vec3 Bezier::evalCubic(const glm::vec3& p0, const glm::vec3& p1, const glm::vec3& p2, const glm::vec3& p3, float t)
{
    float u = 1.0f - t;
    float b0 = u * u * u;
    float b1 = 3.0f * u * u * t;
    float b2 = 3.0f * u * t * t;
    float b3 = t * t * t;
    return p0 * b0 + p1 * b1 + p2 * b2 + p3 * b3;
}

glm::mat4 Bezier::getMatrix() const
{
    if (pts.size() < 4) return glm::mat4(1.0f);

    // poèet segmentù (pøedpoklad: sdílené body => segmentCount = (pts.size()-1)/3)
    int segmentCount = std::max(1, (int)((pts.size() - 1) / 3));
    double now = glfwGetTime();
    double elapsed = (now - startTime) * speed;

    // normalizovat na [0,1]
    float globalT;
    if (loop) {
        globalT = fmod((float)elapsed, 1.0f);
        if (globalT < 0.0f) globalT += 1.0f;
    }
    else {
        globalT = glm::clamp((float)elapsed, 0.0f, 1.0f);
    }

    float segF = globalT * (float)segmentCount;
    int segIndex = std::min(segmentCount - 1, (int)floor(segF));
    float localT = segF - segIndex;

    int base = segIndex * 3;
    glm::vec3 p = evalCubic(pts[base + 0], pts[base + 1], pts[base + 2], pts[base + 3], localT);

    // vrátíme jen translaci; pokud chcete orientaci podle tangenty, spoèítejte derivaci a vytvoøte rotaèní matici
    return glm::translate(glm::mat4(1.0f), p);
}