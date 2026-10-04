build:
	rm -rf main	
	gcc main.c -o main
	./main < test-left-diagonal-1/stdin > test-left-diagonal-1/output 
	./main < test-left-diagonal-2/stdin > test-left-diagonal-2/output
	./main < test-letter-t-1/stdin > test-letter-t-1/output
	./main < test-letter-t-2/stdin > test-letter-t-2/output
	./main < test-rectangle-hole-1/stdin > test-rectangle-hole-1/output
	./main < test-rectangle-hole-2/stdin > test-rectangle-hole-2/output
	./main < test-rectangle-long/stdin > test-rectangle-long/output
	./main < test-rectangle-numbers-1/stdin > test-rectangle-numbers-1/output
	./main < test-rectangle-numbers-2/stdin > test-rectangle-numbers-2/output
	./main < test-rectangle-numbers-3/stdin > test-rectangle-numbers-3/output
	./main < test-rectangle-vertical-numbers/stdin > test-rectangle-vertical-numbers/output
	./main < test-rectangle-wide/stdin > test-rectangle-wide/output
	./main < test-square/stdin > test-square/output
	./main < test-wrong-input/stdin > test-wrong-input/output

	diff test-left-diagonal-1/stdin test-left-diagonal-1/output 
	diff test-left-diagonal-2/stdin test-left-diagonal-2/output
	diff test-letter-t-1/stdin test-letter-t-1/output
	diff test-letter-t-2/stdin test-letter-t-2/output
	diff test-rectangle-hole-1/stdin test-rectangle-hole-1/output
	diff test-rectangle-hole-2/stdin test-rectangle-hole-2/output
	diff test-rectangle-long/stdin test-rectangle-long/output
	diff test-rectangle-numbers-1/stdin test-rectangle-numbers-1/output
	diff test-rectangle-numbers-2/stdin test-rectangle-numbers-2/output
	diff test-rectangle-numbers-3/stdin test-rectangle-numbers-3/output
	diff test-rectangle-vertical-numbers/stdin test-rectangle-vertical-numbers/output
	diff test-rectangle-wide/stdin test-rectangle-wide/output
	diff test-square/stdin test-square/output
	diff test-wrong-input/stdin test-wrong-input/output