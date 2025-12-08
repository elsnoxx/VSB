#pragma once

#include "SpotLight.h"
#include "../Observer/Observer.h"
#include "../Observer/ObservableSubjects.h"
#include "../Camera/Camera.h"

class HeadLight : public SpotLight, public Observer
{
private:
	Camera* camera;
	void update(ObservableSubjects subject) override;
public:
	HeadLight(Camera* camera);
};

