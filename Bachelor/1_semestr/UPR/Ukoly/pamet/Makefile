build:
	rm -rf main
	gcc  main.c -o main -g -Wno-unused-result -Wall -Wextra -Wuninitialized -std=c99 -Werror=vla -lm
	
	./main < tests/test-complex-vert/stdin > tests/test-complex-vert/result
	diff tests/test-complex-vert/stdout tests/test-complex-vert/result
	
		

done:
	./main < tests/test-simple/stdin

	./main < tests/test-cent/stdin

	./main < tests/test-outside-range/stdin

	./main < tests/test-simple/stdin 

	./main < tests/test-cent/stdin

	./main < tests/test-large-range/stdin

	./main < tests/test-complex-vert/stdin

	./main < tests/test-no-elems-vert/stdin > tests/test-no-elems-vert/result
	diff tests/test-no-elems-vert/stdout tests/test-no-elems-vert/result

test:
	
	

	./main < tests/test-simple/stdin > tests/test-simple/result
	diff tests/test-simple/stdout tests/test-simple/result




	./main < tests/test-invalid-mode/stdin > tests/test-invalid-mode/result
	diff tests/test-invalid-mode/stdout tests/test-invalid-mode/result


	./main < tests/test-no-elems-vert/stdin > tests/test-no-elems-vert/result
	diff tests/test-no-elems-vert/stdout tests/test-no-elems-vert/result

	./main < tests/test-no-elems-hor/stdin > tests/test-no-elems-hor/result
	diff tests/test-no-elems-hor/stdout tests/test-no-elems-hor/result

	./main < tests/test-big/stdin > tests/test-big/result
	diff tests/test-big/stdout tests/test-big/result

	./main < tests/test-cent/stdin > tests/test-cent/result
	diff tests/test-cent/stdout tests/test-cent/result

	./main < tests/test-cent/stdin > tests/test-cent/result
	diff tests/test-cent/stdout tests/test-cent/result

git:
	git add .
	git commit -m "ukol done"
	git push -u origin main