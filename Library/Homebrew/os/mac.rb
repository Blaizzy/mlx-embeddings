# typed: false
# frozen_string_literal: true

require "os/mac/version"
require "os/mac/xcode"
require "os/mac/sdk"
require "os/mac/keg"

module OS
  # Helper module for querying system information on macOS.
  module Mac
    extend T::Sig

    module_function

    # rubocop:disable Naming/ConstantName
    # rubocop:disable Style/MutableConstant
    ::MacOS = OS::Mac
    # rubocop:enable Naming/ConstantName
    # rubocop:enable Style/MutableConstant

    raise "Loaded OS::Mac on generic OS!" if ENV["HOMEBREW_TEST_GENERIC_OS"]

    # This can be compared to numerics, strings, or symbols
    # using the standard Ruby Comparable methods.
    sig { returns(Version) }
    def version
      @version ||= full_version.strip_patch
    end

    # This can be compared to numerics, strings, or symbols
    # using the standard Ruby Comparable methods.
    sig { returns(Version) }
    def full_version
      @full_version ||= Version.new((ENV["HOMEBREW_MACOS_VERSION"]).chomp)
    end

    sig { params(version: Version).void }
    def full_version=(version)
      @full_version = Version.new(version.chomp)
      @version = nil
    end

    sig { returns(::Version) }
    def latest_sdk_version
      # TODO: bump version when new Xcode macOS SDK is released
      # NOTE: We only track the major version of the SDK.
      ::Version.new("12")
    end
    private :latest_sdk_version

    sig { returns(String) }
    def preferred_perl_version
      if version >= :big_sur
        "5.30"
      else
        "5.18"
      end
    end

    def languages
      return @languages if @languages

      os_langs = Utils.popen_read("defaults", "read", "-g", "AppleLanguages")
      if os_langs.blank?
        # User settings don't exist so check the system-wide one.
        os_langs = Utils.popen_read("defaults", "read", "/Library/Preferences/.GlobalPreferences", "AppleLanguages")
      end
      os_langs = os_langs.scan(/[^ \n"(),]+/)

      @languages = os_langs
    end

    def language
      languages.first
    end

    sig { returns(String) }
    def active_developer_dir
      @active_developer_dir ||= Utils.popen_read("/usr/bin/xcode-select", "-print-path").strip
    end

    sig { returns(T::Boolean) }
    def sdk_root_needed?
      if MacOS::CLT.installed?
        # If there's no CLT SDK, return false
        return false unless MacOS::CLT.provides_sdk?
        # If the CLT is installed and headers are provided by the system, return false
        return false unless MacOS::CLT.separate_header_package?
      end

      true
    end

    # If a specific SDK is requested:
    #
    #   1. The requested SDK is returned, if it's installed.
    #   2. If the requested SDK is not installed, the newest SDK (if any SDKs
    #      are available) is returned.
    #   3. If no SDKs are available, nil is returned.
    #
    # If no specific SDK is requested, the SDK matching the OS version is returned,
    # if available. Otherwise, the latest SDK is returned.

    def sdk_locator
      if CLT.installed? && CLT.provides_sdk?
        CLT.sdk_locator
      else
        Xcode.sdk_locator
      end
    end

    def sdk(v = nil)
      sdk_locator.sdk_if_applicable(v)
    end

    def sdk_for_formula(f, v = nil, check_only_runtime_requirements: false)
      # If the formula requires Xcode, don't return the CLT SDK
      # If check_only_runtime_requirements is true, don't necessarily return the
      # Xcode SDK if the XcodeRequirement is only a build or test requirement.
      return Xcode.sdk if f.requirements.any? do |req|
        next false unless req.is_a? XcodeRequirement
        next false if check_only_runtime_requirements && req.build? && !req.test?

        true
      end

      sdk(v)
    end

    # Returns the path to an SDK or nil, following the rules set by {sdk}.
    def sdk_path(v = nil)
      s = sdk(v)
      s&.path
    end

    def sdk_path_if_needed(v = nil)
      # Prefer CLT SDK when both Xcode and the CLT are installed.
      # Expected results:
      # 1. On Xcode-only systems, return the Xcode SDK.
      # 2. On Xcode-and-CLT systems where headers are provided by the system, return nil.
      # 3. On CLT-only systems with no CLT SDK, return nil.
      # 4. On CLT-only systems with a CLT SDK, where headers are provided by the system, return nil.
      # 5. On CLT-only systems with a CLT SDK, where headers are not provided by the system, return the CLT SDK.

      return unless sdk_root_needed?

      sdk_path(v)
    end

    # See these issues for some history:
    #
    # - {https://github.com/Homebrew/legacy-homebrew/issues/13}
    # - {https://github.com/Homebrew/legacy-homebrew/issues/41}
    # - {https://github.com/Homebrew/legacy-homebrew/issues/48}
    def macports_or_fink
      paths = []

      # First look in the path because MacPorts is relocatable and Fink
      # may become relocatable in the future.
      %w[port fink].each do |ponk|
        path = which(ponk)
        paths << path unless path.nil?
      end

      # Look in the standard locations, because even if port or fink are
      # not in the path they can still break builds if the build scripts
      # have these paths baked in.
      %w[/sw/bin/fink /opt/local/bin/port].each do |ponk|
        path = Pathname.new(ponk)
        paths << path if path.exist?
      end

      # Finally, some users make their MacPorts or Fink directories
      # read-only in order to try out Homebrew, but this doesn't work as
      # some build scripts error out when trying to read from these now
      # unreadable paths.
      %w[/sw /opt/local].map { |p| Pathname.new(p) }.each do |path|
        paths << path if path.exist? && !path.readable?
      end

      paths.uniq
    end

    sig { params(ids: String).returns(T.nilable(Pathname)) }
    def app_with_bundle_id(*ids)
      path = mdfind(*ids)
             .reject { |p| p.include?("/Backups.backupdb/") }
             .first
      Pathname.new(path) if path.present?
    end

    sig { params(ids: String).returns(T::Array[String]) }
    def mdfind(*ids)
      (@mdfind ||= {}).fetch(ids) do
        @mdfind[ids] = Utils.popen_read("/usr/bin/mdfind", mdfind_query(*ids)).split("\n")
      end
    end

    def pkgutil_info(id)
      (@pkginfo ||= {}).fetch(id) do |key|
        @pkginfo[key] = Utils.popen_read("/usr/sbin/pkgutil", "--pkg-info", key).strip
      end
    end

    sig { params(ids: String).returns(String) }
    def mdfind_query(*ids)
      ids.map! { |id| "kMDItemCFBundleIdentifier == #{id}" }.join(" || ")
    end
  end
end
