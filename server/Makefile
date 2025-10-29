.PHONY: config build format

config:
	./config.sh

build: format
	cmake --build build

format:
	clang-format -i S3InvertedLists.h S3InvertedLists.cpp s3_cache_server.cpp 
