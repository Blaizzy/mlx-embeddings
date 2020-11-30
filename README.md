# Homebrew
[![GitHub release](https://img.shields.io/github/release/Homebrew/brew.svg)](https://github.com/Homebrew/brew/releases)

Features, usage and installation instructions are [summarised on the homepage](https://brew.sh). Terminology (e.g. the difference between a Cellar, Tap, Cask and so forth) is [explained here](https://docs.brew.sh/Formula-Cookbook#homebrew-terminology).

## What Packages Are Available?
1. Type `brew search` for a list.
2. Or visit [formulae.brew.sh](https://formulae.brew.sh) to browse packages online.
3. Or use `brew search --desc <keyword>` to browse packages from the command line.

## More Documentation
`brew help`, `man brew` or check [our documentation](https://docs.brew.sh/).

## Troubleshooting
First, please run `brew update` and `brew doctor`.

Second, read the [Troubleshooting Checklist](https://docs.brew.sh/Troubleshooting).

**If you don't read these it will take us far longer to help you with your problem.**

## Contributing
We'd love you to contribute to Homebrew. First, please read our [Contribution Guide](CONTRIBUTING.md) and [Code of Conduct](https://github.com/Homebrew/.github/blob/HEAD/CODE_OF_CONDUCT.md#code-of-conduct).

We explicitly welcome contributions from people who have never contributed to open-source before: we were all beginners once! We can help build on a partially working pull request with the aim of getting it merged. We are also actively seeking to diversify our contributors and especially welcome contributions from women from all backgrounds and people of colour.

A good starting point for contributing is running `brew audit --strict` with some of the packages you use (e.g. `brew audit --strict wget` if you use `wget`) and then read through the warnings, try to fix them until `brew audit --strict` shows no results and [submit a pull request](https://docs.brew.sh/How-To-Open-a-Homebrew-Pull-Request). If no formulae you use have warnings you can run `brew audit --strict` without arguments to have it run on all packages and pick one.

Alternatively, for something more substantial, check out one of the issues labeled `help wanted` in [Homebrew/brew](https://github.com/homebrew/brew/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22) or [Homebrew/homebrew-core](https://github.com/homebrew/homebrew-core/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22).

Good luck!

## Donations
Homebrew is a non-profit project run entirely by unpaid volunteers. We need your funds to pay for software, hardware and hosting around continuous integration and future improvements to the project. Every donation will be spent on making Homebrew better for our users.

Please consider a regular donation through [GitHub Sponsors](https://github.com/sponsors/Homebrew) or [Patreon](https://www.patreon.com/homebrew).

Alternatively, if you'd rather make a one-off payment:

- [Donate with PayPal](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=V6ZE57MJRYC8L)
- Donate by USA $ check from a USA bank:
  - Make check payable to "Software Freedom Conservancy, Inc." and place "Directed donation: Homebrew" in the memo field. Checks should then be mailed to:
    - Software Freedom Conservancy, Inc.
      137 Montague ST  STE 380
      BROOKLYN, NY 11201             USA
- Donate by wire transfer: contact accounting@sfconservancy.org for wire transfer details.

Homebrew is a member of the [Software Freedom Conservancy](https://sfconservancy.org) which provides us with an ability to receive tax-deductible, Homebrew earmarked donations (and [many other services](https://sfconservancy.org/members/services/)). Software Freedom Conservancy, Inc. is a 501(c)(3) organization incorporated in New York, and donations made to it are fully tax-deductible to the extent permitted by law.

## Security
Please report security issues to our [HackerOne](https://hackerone.com/homebrew/).

## Who We Are
Homebrew's [Project Leader](https://docs.brew.sh/Homebrew-Governance#6-project-leader) is [Mike McQuaid](https://github.com/MikeMcQuaid).

Homebrew's [Project Leadership Committee](https://docs.brew.sh/Homebrew-Governance#4-project-leadership-committee) is [Misty De Meo](https://github.com/mistydemeo), [Shaun Jackman](https://github.com/sjackman), [Jonathan Chang](https://github.com/jonchang), [Sean Molenaar](https://github.com/SMillerDev) and [Markus Reiter](https://github.com/reitermarkus).

Homebrew's [Technical Steering Committee](https://docs.brew.sh/Homebrew-Governance#7-technical-steering-committee) is [Michka Popoff](https://github.com/imichka), [FX Coudert](https://github.com/fxcoudert), [Markus Reiter](https://github.com/reitermarkus), [Misty De Meo](https://github.com/mistydemeo) and [Mike McQuaid](https://github.com/MikeMcQuaid).

Homebrew/brew's Linux maintainers are [Michka Popoff](https://github.com/imichka), [Shaun Jackman](https://github.com/sjackman), [Dawid Dziurla](https://github.com/dawidd6), [Issy Long](https://github.com/issyl0) and [Maxim Belkin](https://github.com/maxim-belkin).

Homebrew's other current maintainers are [Claudia Pellegrino](https://github.com/claui), [Zach Auten](https://github.com/zachauten), [Rui Chen](https://github.com/chenrui333), [Vitor Galvao](https://github.com/vitorgalvao), [Caleb Xu](https://github.com/alebcay), [Gautham Goli](https://github.com/GauthamGoli), [Steven Peters](https://github.com/scpeters), [Bo Anderson](https://github.com/Bo98), [William Woodruff](https://github.com/woodruffw), [Igor Kapkov](https://github.com/igas), [Sam Ford](https://github.com/samford), [Alexander Bayandin](https://github.com/bayandin), [Izaak Beekman](https://github.com/zbeekman), [Eric Knibbe](https://github.com/EricFromCanada), [Viktor Szakats](https://github.com/vszakats), [Thierry Moisan](https://github.com/moisan), [Steven Peters](https://github.com/scpeters), [Tom Schoonjans](https://github.com/tschoonj), [Issy Long](https://github.com/issyl0), [CoreCode](https://github.com/core-code), [Randall](https://github.com/ran-dall), [Rylan Polster](https://github.com/Rylan12), [SeekingMeaning](https://github.com/SeekingMeaning), [William Ma](https://github.com/whoiswillma) and [Dustin Rodrigues](https://github.com/dtrodrigues).

Former maintainers with significant contributions include [Jan Viljanen](https://github.com/javian), [JCount](https://github.com/jcount), [commitay](https://github.com/commitay), [Dominyk Tiller](https://github.com/DomT4), [Tim Smith](https://github.com/tdsmith), [Baptiste Fontaine](https://github.com/bfontaine), [Xu Cheng](https://github.com/xu-cheng), [Martin Afanasjew](https://github.com/UniqMartin), [Brett Koonce](https://github.com/asparagui), [Charlie Sharpsteen](https://github.com/Sharpie), [Jack Nagel](https://github.com/jacknagel), [Adam Vandenberg](https://github.com/adamv), [Andrew Janke](https://github.com/apjanke), [Alex Dunn](https://github.com/dunn), [neutric](https://github.com/neutric), [Tomasz Pajor](https://github.com/nijikon), [Uladzislau Shablinski](https://github.com/vladshablinsky), [Alyssa Ross](https://github.com/alyssais), [ilovezfs](https://github.com/ilovezfs), [Chongyu Zhu](https://github.com/lembacon) and Homebrew's creator: [Max Howell](https://github.com/mxcl).

## Community
- [Homebrew/discussions (forum)](https://github.com/homebrew/discussions/discussions)
- [@MacHomebrew (Twitter)](https://twitter.com/MacHomebrew)

## License
Code is under the [BSD 2-clause "Simplified" License](LICENSE.txt).
Documentation is under the [Creative Commons Attribution license](https://creativecommons.org/licenses/by/4.0/).

## Sponsors
Our macOS continuous integration infrastructure is hosted by [MacStadium's Orka](https://www.macstadium.com/customers/homebrew).

[![Powered by MacStadium](https://cloud.githubusercontent.com/assets/125011/22776032/097557ac-eea6-11e6-8ba8-eff22dfd58f1.png)](https://www.macstadium.com)

Our bottles (binary packages) are hosted by [Bintray](https://bintray.com/homebrew).

[![Downloads by Bintray](https://bintray.com/docs/images/downloads_by_bintray_96.png)](https://bintray.com/homebrew)

Secure password storage and syncing is provided by [1Password for Teams](https://1password.com/teams/).

[![1Password](https://1password.com/img/press/logo.e5bf08b08157e08a1b9112fcd58a54b6.svg)](https://1password.com)

Homebrew is a member of the [Software Freedom Conservancy](https://sfconservancy.org).

[![Software Freedom Conservancy](https://sfconservancy.org/img/conservancy_64x64.png)](https://sfconservancy.org)

Homebrew is generously supported by [Randy Reddig](https://github.com/ydnar) and 439 other users via [GitHub Sponsors](https://github.com/sponsors/Homebrew).
