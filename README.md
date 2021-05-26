The iterative_methods project implements iterative methods and
associated utilities in Rust.

It currently demonstrates the following techniques we find powerful:
- Implement iterative methods as StreamingIterators.
- Implement iterative methods utilities as generic adaptors
  of StreamingIterators.
- Test non-trivial methods via property testing (quickcheck).
- Generic output via streaming yaml

If you're not familiar with iterative methods or what the above mean,
[start
here](https://daniel-vainsencher.github.io/book/iterative_methods_part_1.html).

Future plans:
- Expand/stabilize design
- Add more iterative methods
- Add higher level utilities
- Add simple function call interface to methods.

Stability/evolution:
- The design is actively evolving, breakage is to be expected
  everywhere. Feedback welcome! email us or open issues on the repo.
- Some utilities (e.g., take_until) probably belong elsewhere (e.g.,
  {Streaming}Iterator) and so might migrate entirely.

# Licensing

This project is dual-licensed under the [Apache](LICENSE_APACHE.md)
and [MIT](LICENSE_MIT.md) licenses. You may use this code under the
terms of either license.

Contributing to this repo in any form constitutes agreement to license
any such contributions under all licenses specified in the COPYING
file at that time.
