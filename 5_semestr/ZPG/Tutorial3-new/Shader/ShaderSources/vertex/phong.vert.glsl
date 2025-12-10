#version 450 core

layout(location = 0) in vec3 vp;   // position
layout(location = 1) in vec3 vn;   // normal
layout(location = 2) in vec2 vt;   // uv !!! (opraveno)

uniform mat4 modelMatrix;
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;

out vec3 worldPosition;
out vec3 worldNormal;
out vec2 uv;

void main()
{
    vec4 worldPos4 = modelMatrix * vec4(vp, 1.0);
    worldPosition = worldPos4.xyz;

    worldNormal = normalize(transpose(inverse(mat3(modelMatrix))) * vn);

    uv = vt;

    gl_Position = projectionMatrix * viewMatrix * worldPos4;
}
