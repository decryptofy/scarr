# OBJECTIVE

This guide is meant to serve as a beginner tutorial for getting started with the SCARR framework. SCARR is a framework used for side channel analysis. This beginner guide is adapted from one of the starter notebooks.

# SIDE CHANNEL ANALYSIS

Side channel analysis is a rapidly developing field where analysis on power output during cryptological processes can be used by attackers to elicit information about a system. Processing the raw outputs and looking for statistical differences that aren’t uniformly distributed can be used to extract keys. The scarr project currently supports a few different engines for power trace analysis with more under development.

For an indepth introduction to the topic of side channel analysis, reference this seminal [paper](https://link.springer.com/article/10.1007/s13389-011-0006-y).

# GETTING STARTED

## 1. File format - ZARR

The ZARR file format is the only file format supported by SCARR for more information, refer to the readme for a more detailed explanation of this choice of file format.
    
In brief Zarr is a flexible and efficient array storage format for scientifc data with its primary goal being to overcome environment memory limitations caused by working with large datasets such as in SCA.

Zarr breaks large datasets into small chunks accessible with key-value pairings. This approach allows for more efficient compression when doing random array access and for minimizing the overhead caused by working with large datasets. The format of Zarr makes it seamlessly integrateable with parallel and distributed computing.

## 2. Optional: Using VScode or other IDEs with Jupyter notebooks

If you have familiarity working in an IDE from class, be sure to search for a jupyter extension. This will be useful when running prebuilt jupyter notebooks like the starter notebooks in the [scarr-jupyter directory](https://github.com/decryptofy/scarr-jupyter)

## 3. Quick reference for building a notebook

### A. Untar data
Optional: If your data is stored as a tarball, be sure to untar the data and recover the original directory tree for the data.

    !tar -xvf {file_name}
        
### B. Import required packages

#### 1. Import the engine(s) you wish to use

Currently supported engines are NICV, chi2test, CPA, MIA, MIM, SNR, stats, ttest.

    from scarr.engines.snr import {package} as package

#### 2. Import trace handler

    from scarr.file_handling.trace_handler import TraceHandler as th

#### 3. Import sbox weight function

    from scarr.model_values.sbox_weight import SboxWeight

#### 4. Import containers

    from scar.container.container import Container, ContainerOptions

### C. When calling an engine

#### 1. Create a handler object

    handler = th(fileName={dataset})

#### 2. Create an instance of the model

For engines that take in a model as a parameter

    model = SboxWeight()

#### 3. Create an instance of the engine

    engine = {engine}(params)

#### 4. Create an instance of the container

    container = Container(options=ContainerOptions(engine=engine, handler=handler))

#### 5. Run the container

    engine.run(container)

### D. Handling output

#### 1. Store results to a local variable

    results = engine.get_result()

#### 2. Shape results and plot

Shape the results and plot them. This is done using the pyplot package.

    fig, ax = plt.subplots(figsize=(32, 4))
    ax.plot(results[0,0,:], color='red')

    ax.set_xlabel('Samples')
    ax.set_ylabel('{Engine}')
    plt.show()