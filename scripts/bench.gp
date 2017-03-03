set title 'performance comparison'
set xlabel "N"
set ylabel "time(sec)"
set xrange [100:25000]
set terminal png
set output 'output_plot.png'
plot "result_clock_gettime.csv" using 1:2 w lp pt 7 lc 1 title 'baseline', "result_clock_gettime.csv" using 1:3 w lp pt 7 lc 2 title 'openmp_2', "result_clock_gettime.csv" using 1:4 w lp pt 7 lc 3 title 'openmp_4', "result_clock_gettime.csv" using 1:5 w lp pt 7 lc 5 title 'avx', "result_clock_gettime.csv" using 1:6 w lp pt 7 lc 6 title 'avxunroll'
