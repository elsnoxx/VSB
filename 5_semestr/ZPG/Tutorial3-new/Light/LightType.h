#pragma once

// Simple enum for light kinds used across the renderer/shaders.
// Kept as plain enum for C compatibility with existing code, but could
// be migrated to enum class to improve type safety.
enum LightType {
    DIRECTIONAL = 0,
    POINT = 1,
    SPOT = 2
};