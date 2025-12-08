#version 330 core

in vec3 normalVS;
out vec4 frag_colour;

void main()
{
    vec3 rgb = normalVS * 0.5 + 0.5;
    frag_colour = vec4(rgb, 1.0);
}