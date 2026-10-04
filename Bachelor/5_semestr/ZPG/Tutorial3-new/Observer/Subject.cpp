#include "Subject.h"

void Subject::attach(Observer* observer) {
	observers.push_back(observer);
}


void Subject::notify(ObservableSubjects subject) {
	for (auto* observer : observers) {
		observer->update(subject);
	}
}

void Subject::detach(Observer* observer) {
	for (size_t i = 0; i < observers.size(); ++i) {
		if (observers[i] == observer) {
			observers.erase(observers.begin() + i);
			break;
		}
	}
}
