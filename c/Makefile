.PHONY: clean check test_hilbert test_linear test_s4d test_gelu test_softmax test_tlts test_end_to_end

CC      = gcc
CFLAGS  = -Wall -Wextra -O2 -I./include
LDFLAGS = -lm

main: src/main.c src/nn.c src/hilbert.c src/linear.c src/s4d.c src/gelu.c src/softmax.c src/tlts.c
	$(CC) $(CFLAGS) -o main $^ $(LDFLAGS)

test_hilbert: tests/test_hilbert.c src/hilbert.c
	$(CC) $(CFLAGS) -o test_hilbert $^ $(LDFLAGS)
	./test_hilbert

test_linear: tests/test_linear.c src/linear.c
	$(CC) $(CFLAGS) -o test_linear $^ $(LDFLAGS)
	./test_linear

test_s4d: tests/test_s4d.c src/s4d.c
	$(CC) $(CFLAGS) -o test_s4d $^ $(LDFLAGS)
	./test_s4d

test_gelu: tests/test_gelu.c src/gelu.c
	$(CC) $(CFLAGS) -o test_gelu $^ $(LDFLAGS)
	./test_gelu

test_softmax: tests/test_softmax.c src/softmax.c
	$(CC) $(CFLAGS) -o test_softmax $^ $(LDFLAGS)
	./test_softmax

test_tlts: tests/test_tlts.c src/tlts.c
	$(CC) $(CFLAGS) -o test_tlts $^ $(LDFLAGS)
	./test_tlts

test_end_to_end: tests/test_end_to_end.c src/nn.c src/hilbert.c src/linear.c src/s4d.c src/gelu.c src/softmax.c src/tlts.c
	$(CC) $(CFLAGS) -o test_end_to_end $^ $(LDFLAGS)

check: test_hilbert test_linear test_s4d test_gelu test_softmax test_tlts

clean:
	rm -f main test test_hilbert test_linear test_s4d test_gelu test_softmax test_tlts test_end_to_end
