#version 330
layout(location=0) in vec3 vp;
layout(location=1) in vec3 vc;

uniform mat4 modelMatrix;
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;

out vec3 color;
out vec3 Normal;

void main () {
     color = vc;
     gl_Position = projectionMatrix * viewMatrix * vec4 (vp, 1.0);
}