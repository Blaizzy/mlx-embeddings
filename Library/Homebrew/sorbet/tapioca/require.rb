# typed: strict
# frozen_string_literal: true

# This should not be made a constant or Tapioca will think it is part of a gem.
dependency_require_map = {
  "activesupport" => "active_support",
  "ruby-macho"    => "macho",
}.freeze

# Don't start coverage tracking automatically
ENV["SIMPLECOV_NO_DEFAULTS"] = "1"

Bundler.definition.locked_gems.specs.each do |spec|
  name = spec.name

  # These sorbet gems do not contain any library files
  next if name == "sorbet"
  next if name == "sorbet-static"
  next if name == "sorbet-static-and-runtime"

  name = dependency_require_map[name] if dependency_require_map.key?(name)

  require name
rescue LoadError
  raise unless name.include?("-")

  name = name.tr("-", "/")
  require name
end
