# You have a numerical problem. 

## You applied an iterative method naively, now you have 10 problems that recur forever.

Suppose we have a problem, and initial solution (probably bad) and a
function f such that f(x) is a better solution than x by some numeric
cost measure g. Then applying f repeatedly gives us a sequence of ever
better solutions, hopefully converging to an optimal solution. The
most common initial "I want to see this work!" implementation looks
like:

	x = x0
	for i in 100:
		x = f(x)

But as soon as you've implemented such a method, many additional
considerations come up:
- are 100 iterations enough? 
- are x actually ever improving? 
- how long is each iteration taking? 
- perhaps we want to log every 10th x for later plotting?
- it crashed after 2 days of iterations, perhaps I should have saved
  intermediate iterations...

Following the path of least resistance, we might add `print(g(x))` at
the end, move `print(g(x))` into the loop, use timers and print
differences, etc. In general, we find ourselves adding a few lines of
code, then moving, deleting, re-adding, commenting them out... and on
the next project perhaps copy and pasting them again. 

Some of the time, answering our questions require access to internal
state of f, which therefore gets inlined into the loop, and now all
these external concerns are mixed with the algorithm, possibly adding
bugs to the very algorithm they were supposed to help understand!

Argh! 

## Can't we just write the algorithm once, write each utility once, and somehow do only the wiring together we need? 

Yes, kind of, but it is language specific. LT;DR:

* f should not be a black box that transforms solutions x; it should
  instead take and returns a struct s with all interesting state of
  the algorithm.
* To start the sequence of states, convert a problem p into an initial
  state s0
* Abstract the repeated application of f as a stream of states s,
  assuming your language has one.
* Abstract the utilities as stream adaptors. Such a utility takes a
  stream and applies to each a transformation (e.g., measure value), a
  side effect (log some states) or both.

I did not invent this approach, and will happily add further
references!; briefly, this hope has a long history of programming
language specific ([a Haskell mailing list circa
2006](https://mail.haskell.org/pipermail/haskell-cafe/2006-August/017394.html))
answers. A recent and nice entry tackling this problem in Julia is
"[Iterative methods done right (because life is too short to write
for-loops)](http://lostella.github.io/2018/07/25/iterative-methods-done-right.html)"
(referred to as IM below) proposes to write iterative methods as Julia
iterators and various utilities for them as iterator adaptors. I
recommend reading their exposition, which inspired this project!

# How deep can we follow this path in rust?

Rust seems promising for many reasons. 

* Rust loves abstracting sequences; iterators are a (the?) first class
  citizen in, e.g., for loops.
* In Rust abstraction costs are low, so won't dominate the work even
  when iterations are pretty cheap.
* It is a language essentially optimized for writing high efficiency
  reusable abstractions, so when iterator is not an exact fit, we can
  use a variation.

## The simplest thing and how it fails

In simple cases, Rust iterators will do great. We even have some nice
little adaptors pre-made, like `enumerate` with annotates our state
with its location in the stream, and `take` which stops after the
given iteration number:

    for (i, state) in convert_problem_to_iterator(problem).enumerate().take(4) {
        println!("Iteration {} has state {}", i, state);
    }

# TODO here
* rework the fib example, can we actually output the state instead of
  the next value?
* Understand rustdoc better. Link to the fib code examples? Does it
  make sense to make this whole essay literate code using rustdoc?

And we can go quite far with this direction, as long as state is cheap
to copy. The caveat is because iterators return values, not
references, and so for the iterator to own them and the loop body and
down stream adaptors to have access requires copies. Some might be
optimized away by a [sufficiently smart
compiler](https://wiki.c2.com/?SufficientlySmartCompiler) and Rust is
plenty smart, but the Rusty approach to reliable efficiency is to
minimize copies and make them explicit.

Some unprocessed links:

1. https://docs.rs/streaming-iterator/0.1.5/streaming_iterator/

