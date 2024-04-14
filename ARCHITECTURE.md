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

This directory contains the Container class which is the main driver of Scarr. This class is passed in the file handler, engine, pre and postprocessing, and any other configuration options that you wish to use (i.e. tiles and model values you wish to analyze). One role of this class is to handle user interaction with scarr. A typical user will configure all necessary objects, pass them into a container, tell a container to run its computations, and then pull the subsequent results from the container. A more interior role of the container is to handle the interactions between the pre-processes, filters, file handler, and engine objects. 

### `file_handling`

This directory contains the TraceHandler class which is the file handler of Scarr. This class' main responsibilities are to compute information (i.e. indices/ranges for each batch) required for batching and pass back batches of data that the container class requests.

### `filters`

This directory contains all classes that can be used to filter data during run time. The role of these classes are to shape trace data in a way to aid computations. This may take the form of data shifting (sum of absolute differences) or directly modifying the trace values (normalization). These objects are meant to be able to be chained together to enable enhanced analysis. Filters are applied to traces on a per-batch basis inside of the container before the batch is passed back to the engine.

### `preprocessing`

Currently Scarr does not have any preprocessing enabled. However, we are working on this! Soon we hope to have preprocessing algorithms such as principle component analysis to further enable advanced analysis. These classes will be applied at the begginning of a container run call and will be similar in functionality to the filter classes.

### `engines`

This directory contains all available engines that scarr currently supports. Engines are the main compute class of Scarr. Their main role is to contain the batch-wise algorthims that Scarr uses to compute results. Currently Scarr supports two kinds of Engines, leakage detection (i.e. SNR) and key extraction (i.e. CPA). The first is used as a metric to determine if an implementation is attackable. While the second is an attack to draw cryptographic keys out of said implementation. Any Engine reliant on metadata (i.e. ciphertext, keys, plaintext) for its computation requires the use of a member of the model_values class which passes back values based on said metadata and most times a model of some kind (i.e. Hamming Weight). 

### `model_values`

This directory contains all available metadata based models that Scarr currently supports. These models all currently attack input into the AES-128 S-Box. However, we are actively working on adding more available models. The model_values class is also split into two categories. There are model_values that are purely based on metadata (i.e. plaintext) and model_values that rely on a given key or key-hypothesis (i.e. sbox_weight). Metric Engines are usable with any of the model_values while key-extraction engines are only compatible with model_values that inherit from the GuessModelValue base type.

### `postprocessing`

Currently Scarr does not have any postprocessing enabled. However, we are working on this! Currently all key extraction engines have a get_candidate method that will return the key hypothesis for every computed model_position. 

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
