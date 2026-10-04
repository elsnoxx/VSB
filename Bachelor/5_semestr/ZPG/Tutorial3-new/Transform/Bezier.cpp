// ...existing code...
#include "Bezier.h"
#include <GLFW/glfw3.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/constants.hpp>
#include <cmath>

Bezier::Bezier(const std::vector<glm::vec3>& controlPoints,
               float durationSeconds,
               bool loop_,
               bool orient,
               const glm::vec3& upVec,
               const glm::vec3& pivotOffset_)
    : pts(controlPoints),
      duration(std::max(0.0001f, durationSeconds)),
      loop(loop_),
      orientToTangent(orient),
      up(upVec),
      pivotOffset(pivotOffset_),
      startTime(glfwGetTime())
{
}

glm::vec3 Bezier::evalCubic(const glm::vec3& p0, const glm::vec3& p1, const glm::vec3& p2, const glm::vec3& p3, float t)
{
    float u = 1.0f - t;
    // Cubic Bezier basis (Bernstein polynomials of degree 3):
    // B0 = (1-t)^3
    // B1 = 3*(1-t)^2 * t
    // B2 = 3*(1-t) * t^2
    // B3 = t^3
    // Point on curve: P(t) = p0*B0 + p1*B1 + p2*B2 + p3*B3
    float b0 = u * u * u;
    float b1 = 3.0f * u * u * t;
    float b2 = 3.0f * u * t * t;
    float b3 = t * t * t;
    return p0 * b0 + p1 * b1 + p2 * b2 + p3 * b3;
}

glm::vec3 Bezier::evalDerivativeNumeric(const std::vector<glm::vec3>& pts, int base, float t)
{
    const float eps = 1e-4f;
    float t0 = glm::clamp(t - eps, 0.0f, 1.0f);
    float t1 = glm::clamp(t + eps, 0.0f, 1.0f);
    glm::vec3 p0 = evalCubic(pts[base], pts[base+1], pts[base+2], pts[base+3], t0);
    glm::vec3 p1 = evalCubic(pts[base], pts[base+1], pts[base+2], pts[base+3], t1);
    // Numeric derivative (central difference approximation):
    // P'(t) ≈ (P(t+eps) - P(t-eps)) / (2*eps)  (here eps is clamped at boundaries)
    // This gives the tangent vector direction along the curve at parameter t.
    return (p1 - p0) / (t1 - t0);
}

glm::mat4 Bezier::getMatrix() const
{
    if (pts.size() < 4) return glm::mat4(1.0f);

    int segmentCount = std::max(1, (int)((pts.size() - 1) / 3));
    double now = glfwGetTime();
    double elapsed = now - startTime;

    // normalized position along whole spline [0,1]
    float globalT;
    if (loop) {
        globalT = fmod((float)(elapsed / duration), 1.0f);
        if (globalT < 0.0f) globalT += 1.0f;
    } else {
        globalT = glm::clamp((float)(elapsed / duration), 0.0f, 1.0f);
    }

    float segF = globalT * (float)segmentCount;
    int segIndex = std::min(segmentCount - 1, (int)floor(segF));
    float localT = segF - segIndex;
    int base = segIndex * 3;

    // position on curve (evaluate cubic Bezier for this segment)
    // P = p0*B0 + p1*B1 + p2*B2 + p3*B3 (see evalCubic)
    glm::vec3 p = evalCubic(pts[base + 0], pts[base + 1], pts[base + 2], pts[base + 3], localT);

    // base translation (apply pivot offset here if needed)
    glm::mat4 T = glm::translate(glm::mat4(1.0f), p + pivotOffset);

    if (!orientToTangent) {
        return T;
    }

    // tangent (direction) - numeric derivative of the Bezier curve at localT
    // The derivative gives the velocity vector P'(t) which we use as the forward direction
    // for orienting the transform along the path. Here we compute it numerically
    // using a small offset (central difference) in evalDerivativeNumeric().
    glm::vec3 deriv = evalDerivativeNumeric(pts, base, localT);
    if (glm::length(deriv) < 1e-10f) {
        return T; // degenerate derivative, return translation only
    }

    glm::vec3 forward = glm::normalize(deriv);

    // If your model's forward axis is -Z (common), flip:
    // forward = -forward; // odkomentovat pokud je nutné

    // build right/up basis (world up = this->up)
    glm::vec3 right = glm::normalize(glm::cross(up, forward));
    if (glm::length(right) < 1e-10f) {
        // forward is parallel to up - pick arbitrary right
        right = glm::normalize(glm::cross(glm::vec3(1.0f, 0.0f, 0.0f), forward));
    }
    glm::vec3 up2 = glm::cross(forward, right);

    // build rotation matrix (columns = basis vectors)
    glm::mat4 R(1.0f);
    R[0] = glm::vec4(right,  0.0f);   // column 0
    R[1] = glm::vec4(up2,    0.0f);   // column 1
    R[2] = glm::vec4(forward,0.0f);   // column 2

    // optionally apply small correction if model is rotated in model space
    // e.g. if model forward is -X, rotate R by 90 deg around Y:
    R = R * glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(0,1,0));
    // final transform: translate then rotate
    return T * R;
}
