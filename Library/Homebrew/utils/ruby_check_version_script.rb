#!/usr/bin/env ruby --enable-frozen-string-literal --disable=gems,did_you_mean,rubyopt
# typed: true
# frozen_string_literal: true

HOMEBREW_REQUIRED_RUBY_VERSION = ARGV.first.freeze
raise "No Ruby version passed!" if HOMEBREW_REQUIRED_RUBY_VERSION.to_s.empty?

require "rubygems"

ruby_version = Gem::Version.new(RUBY_VERSION)
# This will only happen if the Ruby is too old anyway.
abort unless ruby_version.respond_to?(:canonical_segments)

homebrew_required_ruby_version = Gem::Version.new(HOMEBREW_REQUIRED_RUBY_VERSION)

ruby_version_major, ruby_version_minor, = ruby_version.canonical_segments
homebrew_required_ruby_version_major, homebrew_required_ruby_version_minor, =
  homebrew_required_ruby_version.canonical_segments

if ruby_version_major != homebrew_required_ruby_version_major ||
   ruby_version_minor != homebrew_required_ruby_version_minor
  abort
end
