# Architecture

This document is a minimalist description of SCARR's architecture and its guiding design principles. This is a good place to start to familiarize yourself with the code base.

## Design Principles

* Make sure that code coupling is reduced to the bare minimum needed
    * Files should be mostly self-contained, if necessary, at the expense of code-redundancy
    * Avoid syntactic sugar and unneeded layers of abstraction
    * Goal is an R&D-friendly framework that is easy to extend by others on a per-file basis
    * This approach also aligns best with the per-file licensing of MPL 2.0
* Aim for I/O parallelism (high-level vs. low-level) and asynchronous pre-fetch
    * High-level parallelism: to access the same .zarr in a non-blocking way from multiple threads
    * Low-level parallelism: to fetch the chunks of a batch in parallel (not sequentially)
        * ZarrV2: currently does *not* support this, chunks are fetched sequentially with f.read()
        * ZarrV3: *might* improve this, but otherwise OS-specific approach needed (e.g., io_uring)
    * Asynchronous prefetch: read the next batch while computation over previous batch takes place
        * If compute time > I/O time then: async prefetch works
        * If compute time < I/O time then: async prefetch fails
    * Simplify scaling by using the buffer cache as opposed to explicitly changing accumulator variables
* Aim for compute parallelism despite limitations of GIL
    * Multiprocessing in Python but chunksize=1 (simply to have another GIL)
    * All other threading using Numba or NumPy (for threading efficiency)

## Bird's Eye View

TBD

## Code Map

TBD

## Cross-Cutting Concerns

TBD

