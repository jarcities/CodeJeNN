CXX = clang++
CXXFLAGS = -std=c++23 -Wall -O3 -march=native
TARGET1 = original_runtime
TARGET2 = newArray_runtime
TARGET3 = newForLoop_runtime
TARGET4 = total_runtime
SRC1 = original_test.cpp
SRC2 = newArray_test.cpp
SRC3 = newForLoop_test.cpp
SRC4 = total_time.cpp
FILE1 = original_runtime.txt
FILE2 = newArray_runtime.txt
FILE3 = newForLoop_runtime.txt
ITER = 40

all: $(TARGET1) $(TARGET2) $(TARGET3)

$(TARGET1): $(SRC1)
	$(CXX) $(CXXFLAGS) -o $(TARGET1) $(SRC1)

$(TARGET2): $(SRC2)
	$(CXX) $(CXXFLAGS) -o $(TARGET2) $(SRC2)

$(TARGET3): $(SRC3)
	$(CXX) $(CXXFLAGS) -o $(TARGET3) $(SRC3)

run: all
	for i in $(shell seq 1 $(ITER)); do ./$(TARGET1) >> $(FILE1); done
	for i in $(shell seq 1 $(ITER)); do ./$(TARGET2) >> $(FILE2); done
	for i in $(shell seq 1 $(ITER)); do ./$(TARGET3) >> $(FILE3); done
	$(CXX) $(CXXFLAGS) -o $(TARGET4) $(SRC4)
	./$(TARGET4)

clear:
	rm -f $(TARGET1) $(TARGET2) $(TARGET3) $(TARGET4) $(FILE1) $(FILE2) $(FILE3)
