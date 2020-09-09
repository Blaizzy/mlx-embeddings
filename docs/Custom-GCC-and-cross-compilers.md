# Custom GCC and Cross Compilers

Homebrew depends on having an up-to-date version of Xcode because it comes with specific versions of build tools, e.g. `clang`. Installing a custom version of GCC or Autotools into your `PATH` has the potential to break lots of compiles so we prefer the Apple or Homebrew-provided compilers. Cross-compilers based on GCC will typically be "keg-only" and therefore not linked into your `PATH` by default, or are prefixed with the target architecture, again to avoid conflicting with Apple or Homebrew compilers.

Rather than merging formulae for either of these cases at this time, we're listing them on this page. If you come up with a formula for a new version of GCC or cross-compiler suite, please link it in here.

- Homebrew provides a `gcc` formula for use with Xcode 4.2+.
- Homebrew provides older GCC formulae, e.g. `gcc@7`
- Homebrew provides some cross-compilers and toolchains, but these are named to avoid clashing with the default tools, e.g. `x86_64-elf-gcc`
- Homebrew provides the LLVM Clang, which is bundled with the `llvm` formula.
- [RISC-V](https://github.com/riscv/homebrew-riscv) provides the RISC-V toolchain including binutils and GCC.
