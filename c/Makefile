# Makefile for S4D galaxy classifier C implementation
# run all commands from inside the c/ directory

# tell make these are not real files -- always run them when asked
.PHONY: clean check test_hilbert test_linear test_s4d

# compiler and flags
CC     = gcc
CFLAGS = -Wall -Wextra -O2 -I./include
LDFLAGS = -lm

# build the main demo program linking all layer source files
main: src/main.c src/hilbert.c src/linear.c src/s4d.c src/gelu.c src/softmax.c src/tlts.c
	$(CC) $(CFLAGS) -o main $^ $(LDFLAGS)

# build and run hilbert scan test
test_hilbert: tests/test_hilbert.c src/hilbert.c
	$(CC) $(CFLAGS) -o test_hilbert $^ $(LDFLAGS)
	./test_hilbert

# build and run linear layer test
test_linear: tests/test_linear.c src/linear.c
	$(CC) $(CFLAGS) -o test_linear $^ $(LDFLAGS)
	./test_linear

# build and run s4d layer test
test_s4d: tests/test_s4d.c src/s4d.c
	$(CC) $(CFLAGS) -o test_s4d $^ $(LDFLAGS)
	./test_s4d

# run every test in sequence
check: test_hilbert test_linear test_s4d

# delete all compiled binaries
clean:
	rm -f main test test_hilbert test_linear test_s4d