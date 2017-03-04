reset
set title 'performance comparison'
set ylabel "time(sec)"
set style fill solid
set terminal png
set output 'output_plot.png'

plot [:][:]"thread_result.csv" using 2:xtic(1) with histogram title 'thread' , \
