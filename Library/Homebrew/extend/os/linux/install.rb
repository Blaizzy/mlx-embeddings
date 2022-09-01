# typed: true
# frozen_string_literal: true

module Homebrew
  module Install
    module_function

    # This is a list of known paths to the host dynamic linker on Linux if
    # the host glibc is new enough.  The symlink_ld_so method will fail if
    # the host linker cannot be found in this list.
    DYNAMIC_LINKERS = %w[
      /lib64/ld-linux-x86-64.so.2
      /lib64/ld64.so.2
      /lib/ld-linux.so.3
      /lib/ld-linux.so.2
      /lib/ld-linux-aarch64.so.1
      /lib/ld-linux-armhf.so.3
      /system/bin/linker64
      /system/bin/linker
    ].freeze
    private_constant :DYNAMIC_LINKERS

    GCC_VERSION_SUFFIX = OS::LINUX_GCC_CI_VERSION.delete_suffix(".0").freeze

    # We link GCC runtime libraries that are not specificaly used for Fortran,
    # which are linked by the GCC formula.  We only use the versioned shared libraries
    # as the other shared and static libraries are only used at build time where
    # GCC can find its own libraries.
    GCC_RUNTIME_LIBS = %w[
      libatomic.so.1
      libgcc_s.so.1
      libgomp.so.1
      libstdc++.so.6
    ].freeze
    private_constant :GCC_RUNTIME_LIBS

    def perform_preinstall_checks(all_fatal: false, cc: nil)
      generic_perform_preinstall_checks(all_fatal: all_fatal, cc: cc)
      symlink_ld_so
      symlink_gcc_libs
    end

    def check_cpu
      return if Hardware::CPU.intel? && Hardware::CPU.is_64_bit?
      return if Hardware::CPU.arm?

      message = "Sorry, Homebrew does not support your computer's CPU architecture!"
      if Hardware::CPU.ppc64le?
        message += <<~EOS
          For OpenPOWER Linux (PPC64LE) support, see:
            #{Formatter.url("https://github.com/homebrew-ppc64le/brew")}
        EOS
      end
      abort message
    end
    private_class_method :check_cpu

    def symlink_ld_so
      brew_ld_so = HOMEBREW_PREFIX/"lib/ld.so"
      return if brew_ld_so.readable?

      ld_so = HOMEBREW_PREFIX/"opt/glibc/lib/ld-linux-x86-64.so.2"
      unless ld_so.readable?
        ld_so = DYNAMIC_LINKERS.find { |s| File.executable? s }
        raise "Unable to locate the system's dynamic linker" unless ld_so
      end

      FileUtils.mkdir_p HOMEBREW_PREFIX/"lib"
      FileUtils.ln_sf ld_so, brew_ld_so
    end
    private_class_method :symlink_ld_so

    def symlink_gcc_libs
      gcc_opt_prefix = HOMEBREW_PREFIX/"opt/#{OS::LINUX_PREFERRED_GCC_FORMULA}"

      GCC_RUNTIME_LIBS.each do |library|
        gcc_library = gcc_opt_prefix/"lib/gcc/#{GCC_VERSION_SUFFIX}/#{library}"
        gcc_library_symlink = HOMEBREW_PREFIX/"lib/#{library}"
        # Skip if the link target doesn't exist.
        next unless gcc_library.readable?

        # Also skip if the symlink already exists.
        next if gcc_library_symlink.readable? && (gcc_library_symlink.readlink == gcc_library)

        odie "#{HOMEBREW_PREFIX}/lib does not exist!" unless (HOMEBREW_PREFIX/"lib").readable?

        FileUtils.ln_sf gcc_library, gcc_library_symlink
      end
    end
    private_class_method :symlink_gcc_libs
  end
end
