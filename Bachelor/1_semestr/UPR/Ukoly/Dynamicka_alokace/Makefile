build:
	rm -rf main
	gcc  main.c -o main -g -Wno-unused-result -Wall -Wextra -Wuninitialized -std=c99 -Werror=vla -lm

	valgrind -q --tool=memcheck --track-origins=no --error-exitcode=1 ./main < tests/test-quantum-entanglement/stdin

	

	./main < tests/test-too-many-turtles/stdin > tests/test-too-many-turtles/result
	diff tests/test-too-many-turtles/stdout tests/test-too-many-turtles/result

	./main < tests/test-through-walls/stdin > tests/test-through-walls/result
	diff tests/test-through-walls/stdout tests/test-through-walls/result
	
	
		
test:
	./main < tests/test-through-walls/stdin > tests/test-through-walls/result
	diff tests/test-through-walls/stdout tests/test-through-walls/result
	

git:
	git add .
	git commit -m "9/11 done net complete"
	git push -u origin main