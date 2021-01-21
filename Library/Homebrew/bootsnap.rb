# typed: ignore
# frozen_string_literal: true

# TODO: make this `typed: true` when HOMEBREW_BOOTSNAP is enabled by
# default and/or we vendor bootsnap and the RBI file.

raise "Needs HOMEBREW_BOOTSNAP!" unless ENV["HOMEBREW_BOOTSNAP"]

require "rubygems"
require "bootsnap"

Bootsnap.setup(
  cache_dir:            "#{ENV["HOMEBREW_TEMP"]}/homebrew-bootsnap",
  development_mode:     false, # TODO: use ENV["HOMEBREW_DEVELOPER"]?,
  load_path_cache:      true,
  autoload_paths_cache: true,
  disable_trace:        true,
  compile_cache_iseq:   true,
  compile_cache_yaml:   true,
)
