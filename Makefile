CC = g++
CFLAGS = -g -Wall -std=c++11


PROG1 = main1
PROG2 = main2
PROG3 = main3
PROG4 = main4

SRCS1 = a.cpp
SRCS2 = b.cpp
SRCS3 = c.cpp
SRCS4 = d.cpp


# based on usage, libaries can be add or remove
OPENCV = `pkg-config opencv --cflags --libs` 
THOR = -lthor
PCL = -lpcl_io -lpcl_visualization -lpcl_common -lpcl_features 
VTK = -I/usr/local/include/vtk -lvtkCommonCore-8.1
BOOST = -lboost_regex -lboost_system
GLOG = -lglog
LIBS = $(OPENCV) $(THOR) $(PCL) $(BOOST) $(GLOG)

$(PROG1):$(SRCS1)
	$(CC) $(CFLAGS) -o $(PROG1) $(SRCS1) $(OPENCV) $(THOR)

$(PROG2):$(SRCS2)
	$(CC) $(CFLAGS) -o $(PROG2) $(SRCS2)  $(OPENCV) $(THOR)

$(PROG3):$(SRCS3)
	$(CC) $(CFLAGS) -o $(PROG3) $(SRCS3)  $(OPENCV) $(THOR)

$(PROG4):$(SRCS4)
	$(CC) $(CFLAGS) -o $(PROG4) $(SRCS4)  $(OPENCV) $(THOR)