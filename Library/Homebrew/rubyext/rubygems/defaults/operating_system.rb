# typed: false
# frozen_string_literal: true

# Fixes universal-ruby getting confused whether to install arm64 or x86_64 macOS versions.
# This can be removed when integrated into Bundler: https://github.com/rubygems/rubygems/pull/5978
module Gem
  # @private
  class Specification
    if /^universal\.(?<arch>.*?)-/ =~ (CROSS_COMPILING || RUBY_PLATFORM)
      local_platform = Platform.local
      if local_platform.cpu == "universal"
        ORIGINAL_LOCAL_PLATFORM = local_platform.to_s.freeze

        local_platform.cpu = if arch == "arm64e" # arm64e is only permitted for Apple system binaries
          "arm64"
        else
          arch
        end

        def extensions_dir
          Gem.default_ext_dir_for(base_dir) ||
            File.join(base_dir, "extensions", ORIGINAL_LOCAL_PLATFORM,
                      Gem.extension_api_version)
        end
      end
    end
  end
end

# This doesn't currently exist in system Ruby but it's safer to check.
orig_file = File.join(RbConfig::CONFIG["rubylibdir"], "rubygems", "defaults", "operating_system")
require orig_file if File.exist?(orig_file)
