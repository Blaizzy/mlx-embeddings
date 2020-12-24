# typed: false
# frozen_string_literal: true

module Superenv
  extend T::Sig

  class << self
    undef bin

    # @private
    def bin
      return unless DevelopmentTools.installed?

      (HOMEBREW_SHIMS_PATH/"mac/super").realpath
    end
  end

  alias x11? x11

  undef homebrew_extra_paths,
        homebrew_extra_pkg_config_paths, homebrew_extra_aclocal_paths,
        homebrew_extra_isystem_paths, homebrew_extra_library_paths,
        homebrew_extra_cmake_include_paths,
        homebrew_extra_cmake_library_paths,
        homebrew_extra_cmake_frameworks_paths,
        determine_cccfg

  def homebrew_extra_paths
    paths = []
    paths << MacOS::XQuartz.bin if x11?
    paths
  end

  # @private
  def homebrew_extra_pkg_config_paths
    paths = \
      ["/usr/lib/pkgconfig", "#{HOMEBREW_LIBRARY}/Homebrew/os/mac/pkgconfig/#{MacOS.version}"]
    paths << "#{MacOS::XQuartz.lib}/pkgconfig" << "#{MacOS::XQuartz.share}/pkgconfig" if x11?
    paths
  end

  def homebrew_extra_aclocal_paths
    paths = []
    paths << "#{MacOS::XQuartz.share}/aclocal" if x11?
    paths
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
    paths << MacOS::XQuartz.include.to_s << "#{MacOS::XQuartz.include}/freetype2" if x11?
    paths << "#{self["HOMEBREW_SDKROOT"]}/System/Library/Frameworks/OpenGL.framework/Versions/Current/Headers"
    paths
  end

  def homebrew_extra_library_paths
    paths = []
    if compiler == :llvm_clang
      paths << "#{self["HOMEBREW_SDKROOT"]}/usr/lib"
      paths << Formula["llvm"].opt_lib.to_s
    end
    paths << MacOS::XQuartz.lib.to_s if x11?
    paths << "#{self["HOMEBREW_SDKROOT"]}/System/Library/Frameworks/OpenGL.framework/Versions/Current/Libraries"
    paths
  end

  def homebrew_extra_cmake_include_paths
    paths = []
    paths << "#{self["HOMEBREW_SDKROOT"]}/usr/include/libxml2" if libxml2_include_needed?
    paths << "#{self["HOMEBREW_SDKROOT"]}/usr/include/apache2" if MacOS::Xcode.without_clt?
    paths << MacOS::XQuartz.include.to_s << "#{MacOS::XQuartz.include}/freetype2" if x11?
    paths << "#{self["HOMEBREW_SDKROOT"]}/System/Library/Frameworks/OpenGL.framework/Versions/Current/Headers"
    paths
  end

  def homebrew_extra_cmake_library_paths
    paths = []
    paths << MacOS::XQuartz.lib.to_s if x11?
    paths << "#{self["HOMEBREW_SDKROOT"]}/System/Library/Frameworks/OpenGL.framework/Versions/Current/Libraries"
    paths
  end

  def homebrew_extra_cmake_frameworks_paths
    paths = []
    paths << "#{self["HOMEBREW_SDKROOT"]}/System/Library/Frameworks" if MacOS::Xcode.without_clt?
    paths
  end

  def determine_cccfg
    s = +""
    # Fix issue with sed barfing on unicode characters on Mountain Lion
    s << "s"
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
    generic_setup_build_environment(formula: formula, cc: cc, build_bottle: build_bottle, bottle_arch: bottle_arch)

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
  end

  def no_weak_imports
    append_to_cccfg "w" if no_weak_imports_support?
  end
end
