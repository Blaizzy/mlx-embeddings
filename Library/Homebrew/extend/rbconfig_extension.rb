# typed: false
# frozen_string_literal: true

macos_version = ENV["HOMEBREW_MACOS_VERSION"][0..4]
macos_sdk = "MacOSX#{macos_version}.sdk"

# Ruby hardcodes what might end up being an incorrect SDK path in some of the
# variables that get used in mkmf.rb.
# This patches them up to use the correct SDK.
RbConfig::CONFIG.each do |k, v|
  next unless v.include?("MacOSX.sdk")

  new_value = v.gsub("MacOSX.sdk", macos_sdk)
  next unless File.exist?(new_value)

  RbConfig::CONFIG[k] = new_value
end
