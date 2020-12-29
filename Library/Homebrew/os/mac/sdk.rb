# typed: true
# frozen_string_literal: true

require "os/mac/version"

module OS
  module Mac
    # Class representing a macOS SDK.
    #
    # @api private
    class SDK
      attr_reader :version, :path, :source

      def initialize(version, path, source)
        @version = version
        @path = Pathname.new(path)
        @source = source
      end
    end

    # Base class for SDK locators.
    #
    # @api private
    class BaseSDKLocator
      class NoSDKError < StandardError; end

      def sdk_for(v)
        path = sdk_paths[v]
        raise NoSDKError if path.nil?

        SDK.new v, path, source
      end

      def all_sdks
        sdk_paths.map { |v, p| SDK.new v, p, source }
      end

      def sdk_if_applicable(v = nil)
        sdk = begin
          if v.blank?
            sdk_for OS::Mac.version
          else
            sdk_for v
          end
        rescue NoSDKError
          latest_sdk
        end
        return if sdk.blank?

        # On OSs lower than 11, whenever the major versions don't match,
        # only return an SDK older than the OS version if it was specifically requested
        return if v.blank? && sdk.version < OS::Mac.version

        sdk
      end

      def source
        nil
      end

      private

      def sdk_prefix
        ""
      end

      def sdk_paths
        @sdk_paths ||= begin
          # Bail out if there is no SDK prefix at all
          if File.directory? sdk_prefix
            paths = {}

            Dir[File.join(sdk_prefix, "MacOSX*.sdk")].each do |sdk_path|
              version = sdk_path[/MacOSX(\d+\.\d+)u?\.sdk$/, 1]
              paths[OS::Mac::Version.new(version)] = sdk_path if version.present?
            end

            # Use unversioned SDK path on Big Sur to avoid issues such as:
            # https://github.com/Homebrew/homebrew-core/issues/67075
            # This creates an entry in `paths` whose key is the OS major version
            sdk_path = Pathname.new("#{sdk_prefix}/MacOSX.sdk")
            sdk_settings = sdk_path/"SDKSettings.json"
            if sdk_settings.exist? &&
               (sdk_settings_string = sdk_settings.read.presence) &&
               (sdk_settings_json = JSON.parse(sdk_settings_string).presence) &&
               (version_string = sdk_settings_json.fetch("Version", nil).presence) &&
               (version = version_string[/(\d+)\./, 1].presence)
              paths[OS::Mac::Version.new(version)] = sdk_path
            end

            paths
          else
            {}
          end
        end
      end

      # NOTE: This returns a versioned SDK path, even on Big Sur
      def latest_sdk
        return if sdk_paths.empty?

        v, path = sdk_paths.max { |(v1, _), (v2, _)| v1 <=> v2 }
        SDK.new v, path, source
      end
    end
    private_constant :BaseSDKLocator

    # Helper class for locating the Xcode SDK.
    #
    # @api private
    class XcodeSDKLocator < BaseSDKLocator
      extend T::Sig

      sig { returns(Symbol) }
      def source
        :xcode
      end

      private

      def sdk_prefix
        @sdk_prefix ||= begin
          # Xcode.prefix is pretty smart, so let's look inside to find the sdk
          sdk_prefix = "#{Xcode.prefix}/Platforms/MacOSX.platform/Developer/SDKs"
          # Finally query Xcode itself (this is slow, so check it last)
          sdk_platform_path = Utils.popen_read(DevelopmentTools.locate("xcrun"), "--show-sdk-platform-path").chomp
          sdk_prefix = File.join(sdk_platform_path, "Developer", "SDKs") unless File.directory? sdk_prefix

          sdk_prefix
        end
      end
    end

    # Helper class for locating the macOS Command Line Tools SDK.
    #
    # @api private
    class CLTSDKLocator < BaseSDKLocator
      extend T::Sig

      sig { returns(Symbol) }
      def source
        :clt
      end

      private

      # While CLT SDKs existed prior to Xcode 10, those packages also
      # installed a traditional Unix-style header layout and we prefer
      # using that.
      # As of Xcode 10, the Unix-style headers are installed via a
      # separate package, so we can't rely on their being present.
      # This will only look up SDKs on Xcode 10 or newer, and still
      # return nil SDKs for Xcode 9 and older.
      def sdk_prefix
        @sdk_prefix ||= begin
          if CLT.provides_sdk?
            "#{CLT::PKG_PATH}/SDKs"
          else
            ""
          end
        end
      end
    end
  end
end
