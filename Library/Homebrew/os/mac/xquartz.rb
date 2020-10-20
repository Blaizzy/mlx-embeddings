# typed: true
# frozen_string_literal: true

module OS
  module Mac
    # Helper module for querying XQuartz information.
    #
    # @api private
    module XQuartz
      extend T::Sig

      module_function

      DEFAULT_BUNDLE_PATH = Pathname("Applications/Utilities/XQuartz.app").freeze
      FORGE_BUNDLE_ID = "org.macosforge.xquartz.X11"
      FORGE_PKG_ID = "org.macosforge.xquartz.pkg"

      PKGINFO_VERSION_MAP = {
        "2.6.34"  => "2.6.3",
        "2.7.4"   => "2.7.0",
        "2.7.14"  => "2.7.1",
        "2.7.28"  => "2.7.2",
        "2.7.32"  => "2.7.3",
        "2.7.43"  => "2.7.4",
        "2.7.50"  => "2.7.5_rc1",
        "2.7.51"  => "2.7.5_rc2",
        "2.7.52"  => "2.7.5_rc3",
        "2.7.53"  => "2.7.5_rc4",
        "2.7.54"  => "2.7.5",
        "2.7.61"  => "2.7.6",
        "2.7.73"  => "2.7.7",
        "2.7.86"  => "2.7.8",
        "2.7.94"  => "2.7.9",
        "2.7.108" => "2.7.10",
        "2.7.112" => "2.7.11",
      }.freeze

      # This returns the version number of XQuartz, not of the upstream X.org.
      # The X11.app distributed by Apple is also XQuartz, and therefore covered
      # by this method.
      def version
        if @version ||= detect_version
          ::Version.new @version
        else
          ::Version::NULL
        end
      end

      def detect_version
        if (path = bundle_path) && path.exist? && (version = version_from_mdls(path))
          version
        else
          version_from_pkgutil
        end
      end

      sig { returns(String) }
      def minimum_version
        # Update this a little later than latest_version to give people
        # time to upgrade.
        "2.7.11"
      end

      # @see https://www.xquartz.org/releases/index.html
      sig { returns(String) }
      def latest_version
        "2.7.11"
      end

      def bundle_path
        # Use the default location if it exists.
        return DEFAULT_BUNDLE_PATH if DEFAULT_BUNDLE_PATH.exist?

        # Ask Spotlight where XQuartz is. If the user didn't install XQuartz
        # in the conventional place, this is our only option.
        MacOS.app_with_bundle_id(FORGE_BUNDLE_ID)
      end

      def version_from_mdls(path)
        version = Utils.popen_read(
          "/usr/bin/mdls", "-raw", "-nullMarker", "", "-name", "kMDItemVersion", path.to_s
        ).strip
        version unless version.empty?
      end

      # Upstream XQuartz *does* have a pkg-info entry, so if we can't get it
      # from mdls, we can try pkgutil. This is very slow.
      def version_from_pkgutil
        str = MacOS.pkgutil_info(FORGE_PKG_ID)[/version: (\d\.\d\.\d+)$/, 1]
        PKGINFO_VERSION_MAP.fetch(str, str)
      end

      def prefix
        @prefix ||= Pathname.new("/opt/X11") if Pathname.new("/opt/X11/lib/libpng.dylib").exist?
      end

      def installed?
        !version.null? && !prefix.nil?
      end

      def outdated?
        return false unless installed?

        version < latest_version
      end

      def bin
        prefix/"bin"
      end

      def include
        prefix/"include"
      end

      def lib
        prefix/"lib"
      end

      def share
        prefix/"share"
      end
    end
  end
end
