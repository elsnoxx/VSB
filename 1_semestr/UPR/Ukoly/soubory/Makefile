build:
	rm -rf main
	gcc ./main.c -o main -g -Wno-unused-result -Wall -Wextra -Werror=uninitialized -Werror=vla -Werror=return-type -lm
	./main foo bar -i -i
	
val:
	valgrind ./main NVDA 200 < test-large/stdin

test:
	rm -rf main
	gcc ./main.c -o main -g -Wno-unused-result -Wall -Wextra -Werror=uninitialized -Werror=vla -Werror=return-type -lm
	python3 run-tests.py ./main
	
git:
	git add .
	git commit -m "finaly done all"
	git push -u origin main