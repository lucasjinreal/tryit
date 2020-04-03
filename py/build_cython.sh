# cython cython_t.py --embed
cython cython_t.py
gcc -O3 cython_t.c -shared -fPIC -I /usr/include/python2.7/ -lpython2.7 -o box.so