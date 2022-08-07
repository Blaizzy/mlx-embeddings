# Linux CI in `homebrew/core`

We currently use Ubuntu 16.04 for bottling in `homebrew/core`.

### Ubuntu versus other distributions

As of 2022, around 77% of our users are using Ubuntu. This is the reason why we have chosen this distribution for our base CI image.
We have been using Ubuntu for CI since version 14.04 with success.
The LTS versions have 5 year support. A new LTS version is released every 2 years.

Our bottles are compatible with other distributions like Debian/CentOS, even when compiled on Ubuntu.

### Past and next versions

We are currently moving our CI to Ubuntu 22.04. This work will probably be done before end of 2022.
Moving from Ubuntu 16.04 to Ubuntu 22.04 (and thus skipping version 18.04 and 20.04) took more time than expected due to other more urgent issues we had to take care of.

We plan to proceed with a more regular update from 2022 on. We aim to use the latest Ubuntu LTS version for our CI.

We will start using the latest Ubuntu LTS version for our CI 3 months after it's release. Ideally the migration to the newest version will be done within 12 months after the LTS release, depending of course on the maintainer bandwidth to carry out such a migration.

| Distribution | Glibc | GCC | Usage |
|---|---|---|---|
| Ubuntu 14.04 | 2.19 | 4 | From 2014 to 2017 |
| Ubuntu 16.04 | 2.23 | 5 | From 2017 to 2022 |
| Ubuntu 22.04 | 2.35 | 11 | From 2022 to 2024 |
| Ubuntu 24.04 | ? | ? | From 2024 to 2026 |

### Why always using the latests version?

Homebrew is a rolling-release package manager. We try to ship the newest things as quickest as possible, on macOS and Linux.

When a formula needs a newer GCC because our host GCC in CI is too old, we need to make that formula depend on a newer Homebrew GCC. All C++ dependents of that formula immediately acquire a dependency on Homebrew GCC as well. While we have taken the steps to make sure this no longer holds up GCC updates, it still creates a maintenance headache. This problem is more likely for formula which are very actively maintained and try to use newer features of C++. We decided that we shouldn't be creating maintenance burdens for formulae which are doing the right thing by staying up to date. It makes a lot of sense for Homebrew maintainers to submit upstream fixes when formulae are not working with newer compilers. It makes a lot less sense for Homebrew maintainers to submit fixes because our host compiler is too old.

Note that `glibc` will need to be installed for more users as their `glibc` version will often be too old: disk space is cheap and we have can handle this situation for our users. This situation will often arise when update to a new LTS version and adoption of the new Ubuntu is still low during the first months. For the same reasons as above: we prefer to stay on the bleeding edge and give our users a gentle nudge to think about updating their OS.
