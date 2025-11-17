---
tags:
  - rust
  - python
---

# 006 Python Binding via Rust

Use Pyo3 to create python binding.

Things to be aware of

- `'static` lifetime for `#[[pyclass]`
- reference counting, I guess it acquire GIL?