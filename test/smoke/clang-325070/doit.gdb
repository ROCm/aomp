set pagination off
set confirm off
b 71
r
interpreter-exec mi "-stack-list-frames --thread 10"
interpreter-exec mi "-stack-list-frames --thread 10"
quit
