#version 450 core

in vec3 worldPosition;
in vec3 worldNormal;
in vec2 uv;

out vec4 frag_colour;

// TEXTURE
uniform sampler2D textureUnitID;
uniform int useTexture;

// LIGHTS
uniform int numLights;
uniform bool lightIsOn[16];
uniform int lightTypes[16];           // 0 = directional, 1 = point, 2 = spot
uniform vec3 lightPositions[16];
uniform vec3 lightDirections[16];
uniform vec3 lightColors[16];
uniform float lightIntensities[16];
uniform float lightCutOffs[16];
uniform float lightOuterCutOffs[16];

uniform vec3 viewPosition;
uniform float shininess;
uniform float ambientStrength;
uniform vec3 materialDiffuse;
uniform vec3 materialSpecular;

void main()
{
    vec3 baseColor = materialDiffuse;

    if (useTexture == 1)
    {
        vec4 t = texture(textureUnitID, uv);
        baseColor = t.rgb;
    }

    vec3 N = normalize(worldNormal);
    vec3 V = normalize(viewPosition - worldPosition);

    vec3 totalDiffuse = vec3(0.0);
    vec3 totalSpec = vec3(0.0);

    int n = clamp(numLights, 0, 16);
    for (int i = 0; i < n; ++i)
    {
        if (!lightIsOn[i]) continue;

        vec3 L;
        float attenuation = 1.0;
        float dist = 0.0;

        if (lightTypes[i] == 0) {
            // Directional light -> direction provided (no attenuation)
            L = normalize(-lightDirections[i]); // lightDirections points from light -> scene
        } else {
            // Point or Spot -> compute vector to light and attenuation by distance
            vec3 Lvec = lightPositions[i] - worldPosition;
            dist = length(Lvec);
            L = normalize(Lvec);
            // basic distance attenuation (can be replaced by per-light constants if available)
            attenuation = 1.0 / (1.0 + 0.09 * dist + 0.032 * dist * dist);
        }

        float diff = max(dot(N, L), 0.0);
        vec3 R = reflect(-L, N);
        float spec = (diff > 0.0) ? pow(max(dot(V, R), 0.0), shininess) : 0.0;

        // SPOT-specific falloff
        float spotEffect = 1.0;
        if (lightTypes[i] == 2)
        {
            vec3 spotDir = normalize(lightDirections[i]);
            float theta = dot(-L, spotDir);
            float inner = lightCutOffs[i];
            float outer = lightOuterCutOffs[i];
            float eps = max(inner - outer, 0.0001);
            spotEffect = clamp((theta - outer) / eps, 0.0, 1.0);
        }

        vec3 lightCol = lightColors[i] * lightIntensities[i] * spotEffect;

        totalDiffuse += attenuation * lightCol * diff;
        totalSpec += attenuation * lightCol * spec;
    }

    vec3 ambient = ambientStrength * baseColor;
    vec3 diffuse = totalDiffuse * baseColor;
    vec3 specular = materialSpecular * totalSpec;

    frag_colour = vec4(ambient + diffuse + specular, 1.0);
}