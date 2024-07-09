build:
	rm -rf main
	gcc main.c -o main
	./main 3 1 < /home/elsnoxx/Desktop/UPR/test/test/exact/stdin 
val:
	valgrind -s ./main 3 1 < /home/elsnoxx/Desktop/UPR/test/test/exact/stdin 
test:
	./main 5 2 < /home/elsnoxx/Desktop/UPR/test/test/empty-recipes/stdin 

git:
	git add .
	git commit -m "test spatny vysledek"
	git push -u origin main