.PHONY: clean
clean:
	rm -rf target
	rm -rf build

.PHONY: config
config:
	./config.sh

.PHONY: build
build:
	cmake --build build

.PHONY: format
format:
	cargo fmt
	clang-format -i src/faiss_s3.cpp include/faiss_s3.h src/s3_inverted_lists.cpp include/s3_inverted_lists.h

.PHONY: test
test:
# TODO: Manually define the tests we are running right now.
	cargo test --test embed_tests