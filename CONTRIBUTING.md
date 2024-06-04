# Contributors Guide

If you are interested in making SCARR better, there are many ways you can contribute. For example, you can:

* Submit Bug Report: If you experience any issues or unexpected phenomena while using SCARR, please complete a bug report on GitHub. More detail will help us solve any issues in a prompt manner.
* Suggest Features: Have an idea for a new feature or enhancement? Please share it with us by opening a GitHub feature request. 
* Provide Feedback: Engage with our community by actively commenting in ongoing discussions, feature requests, or proposals.
* Patch Proposals: We welcome any suggested improvements or feature ideas made via pull requests to the repository. Adherence to our coding standards and providing detail will help us promptly incorporate your contributions.
* Documentation: Help us improve SCARR's documentation by proposing edits, clarification, additions. Your assistance will help improve the usability of the project.
* Support Other Users: Share your expertise by answering questions from other users on our GitHub discussion. Your assistance contributes to a healthy and collaborative community.
* Spread the Word: Help promote SCARR by sharing it with others who may find it useful. Word-of-mouth reccomendations are a great way to increase adoption of SCARR.
* Education: Teach others how to use SCARR effectively; your efforts to contribute to the growth of the user community are essential.

# Contributing Process

* Patches should be submitted in the form of Pull Requests to the SCARR repository on GitHub.
* For bugs and feature Requests, please open an [issue](https://github.com/decryptofy/scarr/issues).
* Within the Python context, we do *not* consider GPL to be an appropriate license, run [liccheck](https://pypi.org/project/liccheck/) to make sure not to accidentally import GPL code.
* For newly added Python source code files, include the Mozilla Public License 2.0 (no copyleft exception) header.
* For newly included Python packages, run liccheck -s license_check.ini to make sure that only license-compatible projects are included (next step: include that as a GitHub action).
* New contributors that add/modify/extend source code files under the MPL 2.0 license should include themselves in the list of contributors in the README.md (i.e., the term contribution and contributor follow the MPL 2.0 license definitions).

Steps to Successful Developer Contribution
1.	Identify an opportunity on our GitHub repository, such as a bug fix, outstanding new feature, or documentation.
2.	Fork the SCARR repository on your GitHub account and create a feature branch that describes the nature of your contribution.
3.	Make your changes to the feature branch, committing them frequently, then push and merge to your forked repository’s main branch.
4.	Submit a pull request to the main branch of SCARR, being as detailed as possible in your description to ensure it is integrated in a timely manner.
5.	Your request will undergo a review by code owners, please address any necessary feedback, after which you will receive approval and can merge your pull request.
6.	Repeat this process, stay involved, and celebrate your successful contributions to SCARR!

Examples of Prior Successful Contributions:
1. Lorem Ipsum
2. Lorem Ipsum
3. Lorem Ipsum

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
