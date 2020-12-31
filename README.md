"[Iterative methods done right (because life is to short to write
for-loops](http://lostella.github.io/2018/07/25/iterative-methods-done-right.html)"
(we refer to it as IM below) shows how to write and compose the
building blocks of iterative methods, by modeling them as Julia
iterables (iterators and iterators adaptors). I loved his exposition
and recommend reading it. This repository implements the same ideas in
Rust.

Background

Rust iterators are treated as first class in for loops, with the easy syntax

	for x in make_iter() {
      // Do stuff with x
	}

Some unprocessed links:

1. https://docs.rs/streaming-iterator/0.1.5/streaming_iterator/
1. https://lukaskalbertodt.github.io/2018/08/03/solving-the-generalized-streaming-iterator-problem-without-gats.html#not-the-solution-we-want-the-crate-streaming-iterator

