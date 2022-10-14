# typed: strict
# frozen_string_literal: true

require "etc"

# Fixes universal-ruby getting confused whether to install arm64 or x86_64 macOS versions.
# https://github.com/rubygems/rubygems/issues/4234
# This can be removed when either:
# - We stop using system Ruby
# - System Ruby is updated with this patch (shipped with Ruby 3.1 or later):
#   https://github.com/ruby/ruby/commit/96ce1d9a0ff64494753ad4730f36a0cd7e7a89e7
# - The Rubygems PR https://github.com/rubygems/rubygems/pull/4238 is merged
#   AND we install a new enough Rubygems which includes the said patch, instead of relying the system's version.
platform = Gem::Platform.local.dup
platform.cpu = Etc.uname[:machine] if platform.os == "darwin" && platform.cpu == "universal"
Gem.platforms[Gem.platforms.index(Gem::Platform.local)] = platform

# This doesn't currently exist in system Ruby but it's safer to check.
orig_file = File.join(RbConfig::CONFIG["rubylibdir"], "rubygems", "defaults", "operating_system")
require orig_file if File.exist?(orig_file)
