This project aims to implement iterative methods and associated utilities in Rust. 

It currently demonstrates the following techniques we find powerful:
- Implement iterative methods as StreamingIterators.
- Implement utilities useful to iterative methods as generic adaptors
  of StreamingIterators.
- Test non-trivial methods via property testing (quickcheck).
- Generic output via streaming yaml

Future plans:
- Expand/stabilize design
- Add new iterative methods
- Add higher level utilities
- Add simple function call interface to methods.

Stability/evolution:
- The design is actively evolving, breakage is to be expected
  everywhere.
- Some utilities (e.g., take_until) probably belong elsewhere (e.g.,
  {Streaming}Iterator) and so might migrate entirely.

# Licensing

This project is dual-licensed under the [Apache](LICENSE_APACHE.md)
and [MIT](LICENSE_MIT.md) licenses. You may use this code under the
terms of either license.

Contributing to this repo in any form constitutes agreement to license
any such contributions under all licenses specified in the COPYING
file at that time.
