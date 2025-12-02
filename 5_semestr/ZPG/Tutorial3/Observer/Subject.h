#pragma once
#include <vector>
#include "Observer.h"

class Subject {
public:
    void addObserver(Observer* observer) {
        observers.push_back(observer);
    }

protected:
    void notify(const Camera& cam) {
        for (auto obs : observers) {
            obs->onCameraChanged(cam);
        }
    }

private:
    std::vector<Observer*> observers;
};
