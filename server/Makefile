.PHONY: config build format

config:
	./config.sh

# Do not run format since ubuntu does not have clang-format by default
build:
	cmake --build build

format:
	clang-format -i S3InvertedLists.h S3InvertedLists.cpp s3_cache_server.cpp 
