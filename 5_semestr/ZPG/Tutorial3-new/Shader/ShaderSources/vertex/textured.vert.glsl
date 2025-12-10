#version 330 core
layout(location=0) in vec3 vp;
layout(location=1) in vec3 vc;
layout(location=2) in vec2 vt;

uniform mat4 modelMatrix;
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;

out vec2 uv; // pass to fragment shader
uniform float w = 1.0;
void main () {
    gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(vp, w);
    uv = vt;
}