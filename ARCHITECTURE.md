# Architecture

The high level architecture of the iterative_methods crate. As a tree structure, the code base looks like this:

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
├── feedback_loop.sh // run this once to watch project files and automatically update targets and provide feedback.
└── pull_request_template.md
```

If you haven't already, please see the README.md document for an introduction to the motivation for this library and what it offers.

The remaining sections will address the contents and structure of each directory of the tree. 


## src

In `src`, `lib.rs` exports iterative methods (via algorithms.rs) and
utilities consisting of iterator adaptors. They currently include
- take_until
- assess
- inspect
- last
- time
- step_by
- write_yaml_documents
- enumerate
- {weighted_}reservoir_sample
and their associated implementations. Unit tests are included. 

## examples

Examples demonstrating different functionality. Currently:

- Conjugate Gradient Method (ConjugateGradient)
- Weighted Reservoir Sampling
- Output to yaml, e.g., for animations

## tests

Some integration tests.
