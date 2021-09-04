wheels:
	docker run --rm -v $(shell pwd):/io konstin2/maturin build --release --strip

html:
	sphinx-build -b html docs build/html
