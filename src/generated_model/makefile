CXX=clang++
CXXFLAGS=-std=c++2b -Wall -O3 -march=native
TARGET=runtimes
SRC=test.cpp
FILE=runtimeOriginal
all:$(TARGET)
$(TARGET):$(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)
run:$(TARGET)
	for i in $$(seq 1 20); do ./$(TARGET) >> $(FILE).txt; done
clean:
	rm -f $(TARGET) $(FILE).txt