#pragma once
#pragma once

class Camera;

class Observer {
public:
    virtual void onCameraChanged(const Camera& camera) = 0;
    virtual ~Observer() = default;
};
