build:
	rm -rf main
	gcc  main.c -o main -g -Wno-unused-result -Wall -Wextra -Wuninitialized -std=c99 -Werror=vla -lm
	./main NVDA 200 < test-large/stdin	> test.html
	
val:
	valgrind ./main NVDA 200 < test-large/stdin

test:
	python3 run-tests.py
	
git:
	git add .
	git commit -m "hotovo bez bonusu"
	git push -u origin main