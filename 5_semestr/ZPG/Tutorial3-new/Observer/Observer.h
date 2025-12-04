#pragma once
#include "ObservableSubjects.h"

class Observer
{
public:
	virtual void update(ObservableSubjects Subject) = 0;
};