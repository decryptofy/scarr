# Contributors Guide

If you are interested in making SCARR better, there are many ways you can contribute. For example, you can:

* Submit a bug report
* Suggest a new feature
* Provide feedback by commenting on feature requests/proposals
* Propose a patch by submitting a pull request
* Suggest or submit documentation improvements
* Answer questions from other users
* Share the software with other users who are interested
* Teach others to use the software

# Contributing Process

TBD :-) This project is at its start, we have not figured this part out yet.

* Patches should be submitted in the form of Pull Requests to the SCARR repository on GitHub.
* For bugs and feature Requests, please open an [issue](https://github.com/decryptofy/scarr/issues).
* Within the Python context, we do *not* consider GPL to be an appropriate license, run [liccheck](https://pypi.org/project/liccheck/) to make sure not to accidentally import GPL code.

# Legal

This is an open source project. Contributions you make to this repository are completely voluntary. When you submit an issue, bug report, question, enhancement, pull request, etc., you are offering your contribution without expectation of payment, you expressly waive any future pay claims against SCARR's maintainers related to your contribution, and you acknowledge that this does not create an obligation on the part of the SCARR maintainers of any kind.

# Licensing Explained

In the following, we would like to briefly explain SCARR's license as this might be an important aspect for future contributors. Since SCARR is licensed under the MPL-2.0-no-copyleft-exception (see FAQ [here](https://www.mozilla.org/en-US/MPL/2.0/FAQ/)), SCARR's code itself must be open-source. However, there are nuanced differences compared to GPL that we consider important.

Permissible:
* Larger works can include SCARR *without* revealing code outside of SCARR (unlike GPL)
* Add closed-source/classified extensions to SCARR, if desired, on a per-file basis (unlike LGPL)

Prohibited:
* Cannot include GPL code into SCARR, to make sure the above remains a free choice
* Cannot adopt from SCARR only by acknowledging its use (to prevent undue commercialization)

When working in the hardware security domain, projects can be of sensitive or classified nature. For us, it is perfectly fine for such scenarios to extend SCARR on a per-file basis and keep these additional files under proprietary/classified license, even when distributing SCARR and these extensions to other entities (e.g., from one organization or government entity to another). We consider this a greater freedom compared to imposing GPL rules onto everyone using/extending this project.

Note: the overwhelming majority of Python projects is licensed under MIT, BSD, or Apache 2.0 license that can be combined with SCARR. We are simply being explicit about not including GPL into SCARR, thereby simplifying the inbound=outbound model.
