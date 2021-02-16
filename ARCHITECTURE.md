# Architecture

This document describes the highest level architecture of the library iterative_methods_done_right_in_rust. As a tree structure, the code base looks like this:

```
iterative_methods_done_right_in_rust
├── src/
│   └── lib.rs
├── examples/
├── tests/
├── Cargo.toml
├── ARCHITECTURE.md
├── LICENSE_APACHE.md
├── LICENSE_MIT.md
├── README.md
├── feedback_loop.sh
└── pull_request_template.md
```

If you haven't already, please see the README.md document for an introduction to the motivation for this library and what it offers.

The remaining sections will address the contents and structure of each directory of the tree. 

## src

In `src` the lib.rs file contains all of the code needed to use the tools in this library. Unit tests are included. Most of the code base consists of iterator adaptors. They currently include
- ReservoirIterable
- StepBy	
- Tee
- TimedIterable
and their associated functions and structs.



## examples

Examples demonstrating the basic functionality are provided. Currently, the examples include
- Fibonnacci
- Conjugate Gradient Method (CGIterable)
- Weighted Reservoir Sampling

## tests

Integration tests are provided.
