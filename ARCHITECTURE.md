# Architecture

This document is a minimalist description of SCARR's architecture and its guiding design principles. This is a good place to start familiarizing yourself with the code base.

## Bird's Eye View

SCARR aims at enabling the following side-channel analysis flow (*work in progress*):

data &rarr; filters &rarr; preprocessing &rarr; engine &rarr; postprocessing &rarr; result

Additionally, we want to support chaining of filters and preprocessing modules, such that multiple steps can be combined easily, e.g., filter1, filter2, trace alignment/synchronization, statistical pre-processing, and only then to start the main processing of the side-channel analysis engine.

* filters
   * normalization
   * Sum of Absolute Differences (SAD)
   * High-/Low-/Band-Pass filter
   * ...
* preprocessing
   * Linear Discriminant Analysis (LDA)
   * Principle Component Analysis (PCA)
   * Centered product
   * ...
* engine
   * Leakage Detection (SNR, TVLA, chi2squared, ...)
   * Key Extraction (CPA, MIA, LRA, ...)
   * Including collision and template attacks
   * ...
* postprocessing
   * Key rank estimator
   * ...
 
Since data fetching is expensive, the focus is on one-pass out-of-core processing (processed data can be larger than memory). To perform any analysis, a 'Container' must be configured which is then executed by its run() method.

## Code Map

This section describes the main code structure of SCARR and its most important directories.

```
├── devtools
├── src
│   └── scarr
│       ├── container
│       ├── file_handling
│       ├── filters
│       ├── preprocessing
│       ├── engines
│       └── model_values
│       ├── postprocessing
├── tests
```

### `container`

TBD

### `file_handling`

TBD

### `filters`

TBD

### `preprocessing`

TBD

### `engines`

TBD

### `model_values`

TBD

### `postprocessing`

TBD

## Design Principles

* Make sure that code coupling is reduced to the bare minimum needed
    * Files should be mostly self-contained, if necessary, at the expense of limited code-redundancy
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

## Cross-Cutting Concerns

TBD

### Testing

TBD

### Documentation
