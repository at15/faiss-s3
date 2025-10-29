.PHONY: config build format docker-build docker-run docker-shell

config:
	./config.sh

# Do not run format since ubuntu does not have clang-format by default
build:
	cmake --build build

format:
	clang-format -i S3InvertedLists.h S3InvertedLists.cpp s3_cache_server.cpp

# Docker commands
docker-build:
	docker build -t faiss-s3-server .

# Run container with access to host services (S3Mock on host:9000)
# --add-host makes host.docker.internal work on Linux (works by default on macOS/Windows)
docker-run:
	docker run -p 9001:9001 \
		--add-host=host.docker.internal:host-gateway \
		-e AWS_ACCESS_KEY_ID=test \
		-e AWS_SECRET_ACCESS_KEY=test \
		-e AWS_REGION=us-east-1 \
		-e AWS_EC2_METADATA_DISABLED=true \
		-e S3_ENDPOINT_URL=http://host.docker.internal:9000 \
		-e AWS_ENDPOINT_URL_S3=http://host.docker.internal:9000 \
		faiss-s3-server

docker-shell:
	docker run -it --rm \
		--add-host=host.docker.internal:host-gateway \
		-e AWS_ACCESS_KEY_ID=test \
		-e AWS_SECRET_ACCESS_KEY=test \
		-e AWS_REGION=us-east-1 \
		-e AWS_EC2_METADATA_DISABLED=true \
		-e S3_ENDPOINT_URL=http://host.docker.internal:9000 \
		-e AWS_ENDPOINT_URL_S3=http://host.docker.internal:9000 \
		faiss-s3-server /bin/bash 
