CC = g++
CFLAGS = -g -Wall
SRCS = b.cpp
PROG = b

SRCS2 = a.cpp
PROG2 = a

SRCS2 = c.cpp
PROG2 = c

# based on usage, libaries can be add or remove
OPENCV = `pkg-config opencv --cflags --libs` 
THOR = -lthor
PCL = -lpcl_io -lpcl_visualization -lpcl_common -lpcl_features 
VTK = -I/usr/local/include/vtk -lvtkCommonCore-8.1
BOOST = -lboost_regex -lboost_system
GLOG = -lglog
LIBS = $(OPENCV) $(THOR) $(PCL) $(BOOST) $(GLOG)

$(PROG):$(SRCS)
	$(CC) $(CFLAGS) -o $(PROG) $(SRCS) $(LIBS)

$(PROG2):$(SRCS2)
	$(CC) $(CFLAGS) -o $(PROG2) $(SRCS2)  $(OPENCV) $(THOR)

$(PROG3):$(SRCS3)
	$(CC) $(CFLAGS) -o $(PROG3) $(SRCS3)  $(OPENCV) $(THOR)