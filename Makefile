CC = gcc
CFLAGS = -O0 -std=gnu99 -Wall -fopenmp -mavx
EXECUTABLE = \
	time_test_baseline time_test_openmp_2 time_test_openmp_4 \
	time_test_avx time_test_avxunroll \
	benchmark_clock_gettime

GIT_HOOKS := .git/hooks/pre-commit

$(GIT_HOOKS):
	@scripts/install-git-hooks
	@echo

default: $(GIT_HOOKS) computepi.o
	$(CC) $(CFLAGS) computepi.o time_test.c -DBASELINE -o time_test_baseline -lm -l OpenCL 
	$(CC) $(CFLAGS) computepi.o time_test.c -DOPENMP_2 -o time_test_openmp_2 -lm -l OpenCL 
	$(CC) $(CFLAGS) computepi.o time_test.c -DOPENMP_4 -o time_test_openmp_4 -lm -l OpenCL 
	$(CC) $(CFLAGS) computepi.o time_test.c -DAVX -o time_test_avx -lm -l OpenCL 
	$(CC) $(CFLAGS) computepi.o time_test.c -DAVXUNROLL -o time_test_avxunroll -lm -l OpenCL 
	$(CC) $(CFLAGS) computepi.o benchmark_clock_gettime.c -o benchmark_clock_gettime -lm -l OpenCL
	$(CC) $(CFLAGS) computepi.o thread_benchmark.c -o thread_benchmark -lm -l OpenCL
	$(CC) $(CFLAGS) computepi.o error.c -o error -lm -l OpenCL

.PHONY: clean default

%.o: %.c
	$(CC) -c $(CFLAGS) $< -o $@ 

check: default
	time ./time_test_baseline
	time ./time_test_openmp_2
	time ./time_test_openmp_4
	time ./time_test_avx
	time ./time_test_avxunroll

gencsv: default
	for i in `seq 100 1000 500000`; do \
		printf "%d " $$i;\
		./benchmark_clock_gettime $$i; \
	done > result_clock_gettime.csv
	gnuplot scripts/bench.gp

thread: default
	for i in 25000; do \
		./thread_benchmark $$i; \
	done > thread_result.csv; \
	gnuplot scripts/thread_bench.gp 

error: default
	for i in `seq 100 100 25000`; do \
		printf "%d " $$i;\
		./error $$i; \
	done > error.csv
	gnuplot scripts/error.gp

clean:
	rm -f $(EXECUTABLE) *.o *.s result_clock_gettime.csv thread_result.csv
