# SCARR

SCARR is a Side-Channel Analysis (SCA) framework written in Python that is optimized for performance over compressed data sets to minimize storage needs. At this early stage, SCARR should be considered experimental, i.e., API changes may happen at any time and without prior notice. In addition, only a limited set of analysis algorithms and pre-processing options is currently implemented, but this will change in the future as we hope to continue SCARR as an active open-source project.

SCARR is mainly intended for educational and research purposes. If as an individual you find SCARR useful, please contribute, give us a shout-out, or consider buying us coffee (this project currently runs on coffee only). If you are an organization and you benefit from this development, please consider making a gift to the [PACIFIC Lab](https://pacificlab.org/) at Oregon State University (led by Vincent Immler) to promote SCARR's continued development.

# Table of Contents

* [SCARR Features](#scarr-features)
* [Install](#install)
* [Usage Warning](#usage-warning)
* [Getting Started with SCARR](#getting-started-with-scarr)
* [SCARR's File Format for Side-Channel Analysis Data](#scarrs-file-format-for-side-channel-analysis-data)
* [Working with Other File Formats](#working-with-other-file-formats)
* [Platform Compatibility](#platform-compatibility)
* [Contributing (inbound=outbound)](#contributing-inboundoutbound)
* [License](#license)
* [Authors](#authors)
* [Copyright](#copyright)
* [Citation](#citation)
* [Acknowledgements](#acknowledgements)

# SCARR Features

SCARR is designed to support the following:

* Fast out-of-core computations (processed data can be larger than available memory)
* Processed data can be int or float (raw oscilloscope data or digitally pre-processed)
* Multiple tiles from EM-measurements are stored in the same data set to identify Regions-of-Interest (ROIs)
* Advanced indexing for fast Trace-of-Interest (TOI) and Point-of-Interest (POI) selections
* Analysis algorithms currently include: SNR, TVLA, CPA, MIA (more to come, [check here](https://github.com/decryptofy/scarr/tree/main/src/scarr/engines))

SCARR also aims at maximizing I/O efficiency, including the asynchronous prefetch of (compressed) data sets.

# Install

SCARR can be installed with pip3 from GitHub directly:

```
pip3 install "git+https://github.com/decryptofy/scarr.git"
```

*Alternatively*, you can clone the repository to also get the most recent versions of the [SCARR Jupyter notebooks](https://github.com/decryptofy/scarr-jupyter):

```
git clone https://github.com/decryptofy/scarr.git
cd scarr
git submodule update --init jupyter
```

Afterwards, you can install SCARR by typing:

```
python3 -m pip install .
```

Please note: for now, the reference OS for SCARR is Ubuntu 22.04 LTS with its default Python 3.10. See [here](https://scientific-python.org/specs/spec-0000/) why we will not support older Python versions.

To make use of [Jupyter notebooks](https://jupyter.org/), you may want to use [VS Code](https://code.visualstudio.com/) and its [Jupyter plugin](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter), or [PyCharm](https://www.jetbrains.com/pycharm/), but any other option to run Jupyter notebooks should work, too.

# Usage Warning

Some computations in SCARR can push your hardware to its limits and beyond. Caution is advised when running sustained compute loads as many consumer-grade hardware is not made for this. Especially for laptops, please take into account the best practices under [USAGE.md](https://github.com/decryptofy/scarr/blob/main/USAGE.md). Additionally, SCARR does not do any memory-checking. When attempting computations that exceed the available memory, then based on OS-level settings, SCARR or other applications may be terminated by the OS, resulting in potential data loss. During heavy computations, it is time for coffee, as you cannot use your computer for anything else but SCARR. See also: [DISCLAIMER.md](https://github.com/decryptofy/scarr/blob/main/DISCLAIMER.md)

# Getting Started with SCARR

After installing SCARR and consideration of the usage warning, please proceed as follows:

* select a Jupyter notebook from the `jupyter` subdirectory or its [corresponding repository](https://github.com/decryptofy/scarr-jupyter).
* determine corresponding example data set(s) and download from Box.com: [click here](https://oregonstate.box.com/s/flpkr969do6v1h5a8qwfw5t49c7ivzgl)
* run Jupyter notebook to use SCARR

***Important note for downloading from Box.com:*** we are currently in the process of finding optimized ways for making the download process more convenient and more reliable. Until then, please avoid downloading whole directories that also have trace files in them, as Box.com will attempt to create a .zip prior to the download (causing a timeout while doing so). Select and download data sets only individually.

# SCARR's File Format for Side-Channel Analysis Data

[Zarr](https://zarr.dev/) is a great file format and we use its DirectoryStore as SCARR's native file format. Each data set is represented by a directory that contains the following basic structure:

* traces:
  * directory.zarr/X/Y/traces
* metadata:
  * directory.zarr/X/Y/ciphertext
  * directory.zarr/X/Y/plaintext
  * (*optional*) directory.zarr/X/Y/key

Traces can be left uncompressed or compressed. A chunking of (5000,1000) is recommended. All metadata is left uncompressed and chunked as (5000,16) for AES128. X and Y are the logical coordinates of EM side-channel measurements. Power measurements use the same structure only with /0/0/ as coordinates for /X/Y/.

We are actively supporting the "Zarr-Python Benchmarking & Performance" group to further speed-up Zarr.

# Working with Other File Formats

SCARR only works with its native format and we have no plans to support other file formats. Should you have previously recorded data in other file formats, then you need to convert these data sets to Zarr. We collect example scripts for this format conversion [here](https://oregonstate.box.com/s/flpkr969do6v1h5a8qwfw5t49c7ivzgl), e.g., to convert separate .npy files to a combined .zarr. These scripts are not actively maintained and not optimized.

# Platform Compatibility

SCARR is developed with High-Performance Computing (HPC) considerations in mind. Optimum performance can rely on many aspects of its configuration and the underlying platform. The default batch size (the number of traces processed in parallel at a given point in time) is 5000. Depending on the platform and chosen analysis, other values between 1000 and 10000 may give better results. Please also take into account the following:

* We recommend CPUs with 8 or more physical (performance) cores, preferably with AVX512
* SCARR is optimized for CPUs with SMT (Hyper-Threading); otherwise, mp.pool parameters are not optimal
* A combination of performance and efficiency cores is not specifically considered in mp.pool either
* Fast, low-latency memory should be used (e.g., DDR5-6400 and CL < 32)
* SCARR should *not* be used on NUMA platforms as this degrades performance in unexpected ways
* SCARR is designed to run on Linux/Unix; Windows may work but is *not* supported
* *ulimits need to be adjusted when processing many tiles/byte-positions at the same time*

# Contributing (inbound=outbound)

We want to keep this a no-nonsense project and promote contributions, while minimizing risks to the well-being of the project. If you would like to contribute bug fixes, improvements, and new features back to SCARR, please take a look at our [Contributor Guide](https://github.com/decryptofy/scarr/blob/main/CONTRIBUTING.md) to see how you can participate in this open source project.

Consistent with Section D.6. of the [GitHub Terms of Service](https://docs.github.com/en/site-policy/github-terms/github-terms-of-service) as of November 16, 2020, and the [Mozilla Public License, v. 2.0.](https://www.mozilla.org/en-US/MPL/2.0/), the project maintainer for this project accepts contributions using the inbound=outbound model. When you submit a pull request to this repository (inbound), you are agreeing to license your contribution under the same terms as specified under [License](https://github.com/decryptofy/scarr/blob/main/README.md#license) (outbound).

Note: this is modeled after the terms for contributing to [Ghidra](https://github.com/NationalSecurityAgency/ghidra/blob/master/CONTRIBUTING.md). Our reasoning for this licensing is explained [here](https://github.com/decryptofy/scarr/blob/main/CONTRIBUTING.md#licensing-explained).

# License

This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
This Source Code Form is "Incompatible With Secondary Licenses", as defined by the Mozilla Public License, v. 2.0.

# Authors

SCARR was initiated and designed by Vincent Immler out of a necessity to support his teaching and research at Oregon State University. Under his guidance, two undergraduate students at Oregon State University, Jonah Bosland and Stefan Ene, developed the majority of the initial implementation during the summer 2023. Peter Baumgartner helped us with the testing and analysis on NUMA platforms.

Additional contributions by (new contributors, add yourself here):
* Matt Ruff
* Kevin Yuan
* Alexander Merino
* Tristan Long
* Kayla Barton
* Christopher McLean
* Alana Kihn
* Jackson Hart

# Copyright

Copyright for SCARR (2023-2025) by Vincent Immler.

# Citation

If you use SCARR in your research, please cite our paper:
"High-Performance Design Patterns and File Formats for Side-Channel Analysis" by Jonah Bosland, Stefan Ene, Peter Baumgartner, Vincent Immler.
IACR Transactions on Cryptographic Hardware and Embedded Systems, 2024(2), 769–794.

DOI: [click here](https://doi.org/10.46586/tches.v2024.i2.769-794)

# Acknowledgements

Jonah Bosland has been funded through the Office of Naval Research (ONR) Workforce Development grant (N000142112552) during June-November 2023. Stefan Ene has been
funded through the Summer 2023 Research Experience for Undergraduates (REU) program by the School of EECS at Oregon State University.
