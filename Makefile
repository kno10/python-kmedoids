wheels:
	docker run --rm -v $(shell pwd):/io konstin2/maturin build --release --strip
