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

  def check_dylibs(rebuild_cache:)
    generic_check_dylibs(rebuild_cache: rebuild_cache)

    # glibc and gcc are implicit dependencies.
    # No other linkage to system libraries is expected or desired.
    @unwanted_system_dylibs = @system_dylibs.reject do |s|
      SYSTEM_LIBRARY_ALLOWLIST.include? File.basename(s)
    end
    @undeclared_deps -= [CompilerSelector.preferred_gcc, "glibc"]
  end
end
