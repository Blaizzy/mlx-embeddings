# typed: false
# frozen_string_literal: true

module Superenv
  extend T::Sig

  class << self
    # The location of Homebrew's shims on macOS.
    def shims_path
      HOMEBREW_SHIMS_PATH/"mac/super"
    end

    undef bin

    # @private
    def bin
      return unless DevelopmentTools.installed?

      shims_path.realpath
    end
  end

  undef homebrew_extra_pkg_config_paths,
        homebrew_extra_isystem_paths, homebrew_extra_library_paths,
        homebrew_extra_cmake_include_paths,
        homebrew_extra_cmake_library_paths,
        homebrew_extra_cmake_frameworks_paths,
        determine_cccfg

  # @private
  def homebrew_extra_pkg_config_paths
    ["/usr/lib/pkgconfig", "#{HOMEBREW_LIBRARY}/Homebrew/os/mac/pkgconfig/#{MacOS.version}"]
  end

  # @private
  sig { returns(T::Boolean) }
  def libxml2_include_needed?
    return false if deps.any? { |d| d.name == "libxml2" }
    return false if Pathname("#{self["HOMEBREW_SDKROOT"]}/usr/include/libxml").directory?

    true
  end

  def homebrew_extra_isystem_paths
    paths = []
    paths << "#{self["HOMEBREW_SDKROOT"]}/usr/include/libxml2" if libxml2_include_needed?
    paths << "#{self["HOMEBREW_SDKROOT"]}/usr/include/apache2" if MacOS::Xcode.without_clt?
    paths << "#{self["HOMEBREW_SDKROOT"]}/System/Library/Frameworks/OpenGL.framework/Versions/Current/Headers"
    paths
  end

  def homebrew_extra_library_paths
    paths = []
    if compiler == :llvm_clang
      paths << "#{self["HOMEBREW_SDKROOT"]}/usr/lib"
      paths << Formula["llvm"].opt_lib.to_s
    end
    paths << "#{self["HOMEBREW_SDKROOT"]}/System/Library/Frameworks/OpenGL.framework/Versions/Current/Libraries"
    paths
  end

  def homebrew_extra_cmake_include_paths
    paths = []
    paths << "#{self["HOMEBREW_SDKROOT"]}/usr/include/libxml2" if libxml2_include_needed?
    paths << "#{self["HOMEBREW_SDKROOT"]}/usr/include/apache2" if MacOS::Xcode.without_clt?
    paths << "#{self["HOMEBREW_SDKROOT"]}/System/Library/Frameworks/OpenGL.framework/Versions/Current/Headers"
    paths
  end

  def homebrew_extra_cmake_library_paths
    ["#{self["HOMEBREW_SDKROOT"]}/System/Library/Frameworks/OpenGL.framework/Versions/Current/Libraries"]
  end

  def homebrew_extra_cmake_frameworks_paths
    paths = []
    paths << "#{self["HOMEBREW_SDKROOT"]}/System/Library/Frameworks" if MacOS::Xcode.without_clt?
    paths
  end

  def determine_cccfg
    s = +""
    # Fix issue with >= Mountain Lion apr-1-config having broken paths
    s << "a"
    s.freeze
  end

  # @private
  def setup_build_environment(formula: nil, cc: nil, build_bottle: false, bottle_arch: nil, testing_formula: false)
    sdk = formula ? MacOS.sdk_for_formula(formula) : MacOS.sdk
    if MacOS.sdk_root_needed? || sdk&.source == :xcode
      Homebrew::Diagnostic.checks(:fatal_setup_build_environment_checks)
      self["HOMEBREW_SDKROOT"] = sdk.path

      self["HOMEBREW_DEVELOPER_DIR"] = if sdk.source == :xcode
        MacOS::Xcode.prefix
      else
        MacOS::CLT::PKG_PATH
      end
    else
      self["HOMEBREW_SDKROOT"] = nil
      self["HOMEBREW_DEVELOPER_DIR"] = nil
    end
    generic_setup_build_environment(
      formula: formula, cc: cc, build_bottle: build_bottle,
      bottle_arch: bottle_arch, testing_formula: testing_formula
    )

    # Filter out symbols known not to be defined since GNU Autotools can't
    # reliably figure this out with Xcode 8 and above.
    if MacOS.version == "10.12" && MacOS::Xcode.version >= "9.0"
      %w[fmemopen futimens open_memstream utimensat].each do |s|
        ENV["ac_cv_func_#{s}"] = "no"
      end
    elsif MacOS.version == "10.11" && MacOS::Xcode.version >= "8.0"
      %w[basename_r clock_getres clock_gettime clock_settime dirname_r
         getentropy mkostemp mkostemps timingsafe_bcmp].each do |s|
        ENV["ac_cv_func_#{s}"] = "no"
      end

      ENV["ac_cv_search_clock_gettime"] = "no"

      # works around libev.m4 unsetting ac_cv_func_clock_gettime
      ENV["ac_have_clock_syscall"] = "no"
    end

    # The tools in /usr/bin proxy to the active developer directory.
    # This means we can use them for any combination of CLT and Xcode.
    self["HOMEBREW_PREFER_CLT_PROXIES"] = "1"

    # Deterministic timestamping.
    # This can work on older Xcode versions, but they contain some bugs.
    # Notably, Xcode 10.2 fixes issues where ZERO_AR_DATE affected file mtimes.
    # Xcode 11.0 contains fixes for lldb reading things built with ZERO_AR_DATE.
    self["ZERO_AR_DATE"] = "1" if MacOS::Xcode.version >= "11.0" || MacOS::CLT.version >= "11.0"
  end

  def no_weak_imports
    append_to_cccfg "w" if no_weak_imports_support?
  end
end
