build:
	rm -rf main
	gcc  main.c -o main -g -Wno-unused-result -Wall -Wextra -Wuninitialized -std=c99 -Werror=vla -lm

	./main < test-simple/stdin 
	
test:
	bash run-tests.sh
	
git:
	git add .
	git commit -m "done s rozdelenim"
	git push -u origin main