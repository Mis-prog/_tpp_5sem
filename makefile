all:row

row:
	pgcc -acc -ta=nvidia -fast -Minfo=accel src/sum_row.c -o bin/sum_row
matrix:
	pgcc -acc -ta=nvidia -fast -Minfo=accel src/mult_matrix.c -o bin/mult_matrix
