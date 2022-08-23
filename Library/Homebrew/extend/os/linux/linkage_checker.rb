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
    return unless @libcrypt_found

    # Steps when removing this entirely:
    # - Remove the `display_` overrides here and the associated generic aliases in HOMEBREW_LIBRARY/linkage_checker.rb
    # - Remove the setting of `@libcrypt_found` in `check_dylibs` below.
    odisabled "linkage to libcrypt.so.1", "libcrypt.so.2 in the libxcrypt formula"
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
    generic_broken_library_linkage?(test: test, strict: strict)
  end

  private

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
