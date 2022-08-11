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
    libcrypt.so.1
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
    return unless @libcrypt_found

    # Steps when moving this to `odisabled`:
    # - Remove `libcrypt.so.1` from SYSTEM_LIBRARY_ALLOWLIST above.
    # - Remove the `disable` and `disable_for_developer` kwargs here.
    # - Remove `broken_library_linkage?` override below and the generic alias in HOMEBREW_LIBRARY/linkage_checker.rb.
    # - Remove `fail_on_libcrypt1?`.
    # Steps when removing this entirely (assuming the above has already been done):
    # - Remove the `display_` overrides here and the associated generic aliases in HOMEBREW_LIBRARY/linkage_checker.rb
    # - Remove the setting of `@libcrypt_found` in `check_dylibs` below.
    odeprecated "linkage to libcrypt.so.1", "libcrypt.so.2 in the libxcrypt formula",
                disable:                fail_on_libcrypt1?(strict: strict),
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

  def broken_library_linkage?(strict: false)
    generic_broken_library_linkage?(strict: strict) || (fail_on_libcrypt1?(strict: strict) && @libcrypt_found)
  end

  private

  def fail_on_libcrypt1?(strict:)
    strict || ENV["HOMEBREW_DISALLOW_LIBCRYPT1"].present?
  end

  def check_dylibs(rebuild_cache:)
    generic_check_dylibs(rebuild_cache: rebuild_cache)

    @libcrypt_found = true if @system_dylibs.any? { |s| File.basename(s) == "libcrypt.so.1" }

    # glibc and gcc are implicit dependencies.
    # No other linkage to system libraries is expected or desired.
    @unwanted_system_dylibs = @system_dylibs.reject do |s|
      SYSTEM_LIBRARY_ALLOWLIST.include? File.basename(s)
    end
    # FIXME: Remove this when these dependencies are injected correctly (e.g. through `DependencyCollector`)
    # See discussion at
    #   https://github.com/Homebrew/brew/pull/13577
    @undeclared_deps -= [CompilerSelector.preferred_gcc, "glibc", "gcc"]
  end
end
