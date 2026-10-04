build:
	rm -rf main
	gcc  main.c -o main -g -Wno-unused-result -Wall -Wextra -Wuninitialized -std=c99 -Werror=vla -lm
	./main images/img1.tga out-1.tga font-images < /home/elsnoxx/Desktop/UPR/Ukoly/TGA/test-1/stdin
val: 
	valgrind ./main images/img1.tga out-1.tga font-images < /home/elsnoxx/Desktop/UPR/Ukoly/TGA/test-1/stdin

test:
	rm -rf main
	gcc  main.c -o main -g -Wno-unused-result -Wall -Wextra -Wuninitialized -std=c99 -Werror=vla -lm
	python3 run-tests.py ./main
	
git:
	git add .
	git commit -m "finaly done full"
	git push -u origin main