# typed: true
# frozen_string_literal: true

require "compilers"

class LinkageChecker
  # Libraries provided by glibc and gcc.
  SYSTEM_LIBRARY_ALLOWLIST = %w[
    ld-linux-x86-64.so.2
    libanl.so.1
    libatomic.so.1
    libc.so.6
    libdl.so.2
    libm.so.6
    libmvec.so.1
    libnsl.so.1
    libnss_files.so.2
    libpthread.so.0
    libresolv.so.2
    librt.so.1
    libthread_db.so.1
    libutil.so.1
    libgcc_s.so.1
    libgomp.so.1
    libstdc++.so.6
  ].freeze

  def display_deprecated_warning(strict: false)
    # Steps when moving these to `odisabled`:
    # - Remove the old library from SYSTEM_LIBRARY_ALLOWLIST above.
    # - Remove the `disable` and `disable_for_developer` kwargs here.
    # - Adjust the `broken_library_linkage?` override below to not check for the library.
    # - Remove the relevant `fail_on_lib*?`.
    # If there's no more deprecations left:
    # - Remove the `broken_library_linkage?` override and the generic alias in HOMEBREW_LIBRARY/linkage_checker.rb.
    #
    # Steps when removing handling for a library entirely (assuming the steps to `odisabled` has already been done):
    # - Remove the relevant setting of `@lib*_found` in `check_dylibs` below.
    # - Remove the `odisabled` line
    # If there's no library deprecated/disabled handling left:
    # - Remove the `display_` overrides here and the associated generic aliases in HOMEBREW_LIBRARY/linkage_checker.rb

    odisabled "linkage to libcrypt.so.1", "libcrypt.so.2 in the libxcrypt formula" if @libcrypt_found

    return unless @libnsl_found

    odeprecated "linkage to libnsl.so.1", "libnsl.so.3 in the libnsl formula",
                disable:                fail_on_libnsl1?(strict: strict),
                disable_for_developers: false
  end

  def display_normal_output
    generic_display_normal_output
    display_deprecated_warning
  end

  def display_test_output(puts_output: true, strict: false)
    generic_display_test_output(puts_output: puts_output, strict: strict)
    display_deprecated_warning(strict: strict)
  end

  def broken_library_linkage?(test: false, strict: false)
    generic_broken_library_linkage?(test: test, strict: strict) || (fail_on_libnsl1?(strict: strict) && @libnsl_found)
  end

  private

  def fail_on_libnsl1?(strict:)
    strict || ENV["HOMEBREW_DISALLOW_LIBNSL1"].present?
  end

  def check_dylibs(rebuild_cache:)
    generic_check_dylibs(rebuild_cache: rebuild_cache)

    @libcrypt_found = true if @system_dylibs.any? { |s| File.basename(s) == "libcrypt.so.1" }
    @libnsl_found = true if @system_dylibs.any? { |s| File.basename(s) == "libnsl.so.1" }

    # glibc and gcc are implicit dependencies.
    # No other linkage to system libraries is expected or desired.
    @unwanted_system_dylibs = @system_dylibs.reject do |s|
      SYSTEM_LIBRARY_ALLOWLIST.include? File.basename(s)
    end

    # We build all formulae with an RPATH that includes the gcc formula's runtime lib directory.
    # See: https://github.com/Homebrew/brew/blob/e689cc07/Library/Homebrew/extend/os/linux/extend/ENV/super.rb#L53
    # This results in formulae showing linkage with gcc whenever it is installed, even if no dependency is declared.
    # See discussions at:
    #   https://github.com/Homebrew/brew/pull/13659
    #   https://github.com/Homebrew/brew/pull/13796
    # TODO: Find a nicer way to handle this. (e.g. examining the ELF file to determine the required libstdc++.)
    @undeclared_deps.delete("gcc")
  end
end
