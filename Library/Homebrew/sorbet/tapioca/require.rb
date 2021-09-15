# typed: strict
# frozen_string_literal: true

# This should not be made a constant or Tapioca will think it is part of a gem.
dependency_require_map = {
  "activesupport" => "active_support",
  "ruby-macho"    => "macho",
}.freeze

Bundler.definition.locked_gems.specs.each do |spec|
  name = spec.name

  # sorbet(-static) gem contains executables rather than a library
  next if name == "sorbet"
  next if name == "sorbet-static"

  name = dependency_require_map[name] if dependency_require_map.key?(name)

  require name
rescue LoadError
  raise unless name.include?("-")

  name = name.tr("-", "/")
  require name
end
