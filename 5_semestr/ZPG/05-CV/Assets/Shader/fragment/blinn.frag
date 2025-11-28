#version 330 core

in VS_OUT {
    vec3 worldPos;
    vec3 worldNormal;
} fs_in;

out vec4 fragColor;

// Object material
uniform vec3 uAlbedo       = vec3(0.8); // base color
uniform float uShininess   = 32.0;      // specular exponent

// Light properties
uniform vec3 uLightPos;
uniform vec3 uLightColor   = vec3(1.0);
uniform float uLightIntensity = 1.0;

// Camera
uniform vec3 uCamPos;

void main()
{
    // Normal, light, view
    vec3 N = normalize(fs_in.worldNormal);
    vec3 L = normalize(uLightPos - fs_in.worldPos);
    vec3 V = normalize(uCamPos - fs_in.worldPos);
    vec3 H = normalize(L + V);   // Blinn–Phong half-vector

    // Diffuse term
    float diff = max(dot(N, L), 0.0);

    // Specular term
    float spec = pow(max(dot(N, H), 0.0), uShininess);

    // Lighting components
    vec3 ambient  = 0.1 * uAlbedo;  // Ambient based on material color
    vec3 diffuse  = uAlbedo * uLightColor * (uLightIntensity * diff);
    vec3 specular = uLightColor * (uLightIntensity * spec);

    vec3 finalColor = ambient + diffuse + specular;
    fragColor = vec4(finalColor, 1.0);
}
