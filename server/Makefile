.PHONY: config build format \
        docker-build docker-run docker-shell \
        ci-docker-build-base ci-docker-build-app ci-docker-build-app-from-ghcr \
        ci-docker-validate ci-docker-test-local

config:
	./config.sh

# Do not run format since ubuntu does not have clang-format by default
build:
	cmake --build build

format:
	clang-format -i S3InvertedLists.h S3InvertedLists.cpp s3_cache_server.cpp

# Local Docker commands (no GHCR required)
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

# CI Docker commands (for testing CI build flow locally or in GitHub Actions)
ci-docker-build-base:
	@echo "Building CI base image (dependencies only)..."
	docker build -f ci/Dockerfile.base -t faiss-s3-ci-base:latest .

ci-docker-build-app:
	@echo "Building CI app image (using local base)..."
	docker build -f ci/Dockerfile.app -t faiss-s3-server:latest \
		--build-arg BASE_IMAGE=faiss-s3-ci-base:latest .

ci-docker-build-app-from-ghcr:
	@echo "Building CI app image (using GHCR base)..."
	docker build -f ci/Dockerfile.app -t faiss-s3-server:latest \
		--build-arg BASE_IMAGE=ghcr.io/at15/faiss-s3/ci-base:latest .

ci-docker-validate:
	@echo "Validating Docker dependency sync..."
	./ci/validate-sync.sh

ci-docker-test-local:
	@echo "Testing full CI build flow locally..."
	$(MAKE) ci-docker-build-base
	$(MAKE) ci-docker-build-app
	@echo "Testing app image..."
	docker run --rm faiss-s3-server:latest /app/s3_cache_server --help || true
	@echo "✓ CI build flow test complete" 
