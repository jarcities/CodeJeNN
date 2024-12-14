CXX = clang++
CXXFLAGS = -std=c++2b -Wall -O3 -march=native
TARGET1 = original_runtime
TARGET2 = newArray_runtime
TARGET3 = total_time
SRC1 = original_test.cpp
SRC2 = newArray_test.cpp
SRC3 = total_time.cpp
FILE1 = original_runtime
FILE2 = newArray_runtime
ITER = 40

all: $(TARGET1) $(TARGET2)

$(TARGET1): $(SRC1)
	$(CXX) $(CXXFLAGS) -o $(TARGET1) $(SRC1)

$(TARGET2): $(SRC2)
	$(CXX) $(CXXFLAGS) -o $(TARGET2) $(SRC2)

run: all
	for i in $(shell seq 1 $(ITER)); do ./$(TARGET1) >> $(FILE1).txt; done
	for i in $(shell seq 1 $(ITER)); do ./$(TARGET2) >> $(FILE2).txt; done
	$(CXX) $(CXXFLAGS) -o $(TARGET3) $(SRC3)
	./$(TARGET3)

clean:
	rm -f $(TARGET1) $(TARGET2) $(TARGET3) $(FILE1).txt $(FILE2).txt
