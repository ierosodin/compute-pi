set title 'error test'
set xlabel "N" 
set ylabel "time(sec)"
set xrange [100:25000]
set logscale y
set terminal png
set output 'output_plot.png'
plot "error.csv" using 1:2 w lp pt 7 lc 1 title 'baseline', "error.csv" using 1:3 w lp pt 7 lc 2 title 'openmp_2', "error.csv" using 1:4 w lp pt 7 lc 3 title 'openmp_4', "error.csv" using 1:5 w lp pt 7 lc 4 title 'avx', "error.csv" using 1:6 w lp pt 7 lc 5 title 'avxunroll'
