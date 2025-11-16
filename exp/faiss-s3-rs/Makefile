.PHONY: config
config:
	./config.sh

.PHONY: build
build:
	cmake --build build

.PHONY: format
format:
	cargo fmt
	clang-format -i src/faiss_s3.cpp include/faiss_s3.h