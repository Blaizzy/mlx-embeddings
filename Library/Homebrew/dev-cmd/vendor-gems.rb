# frozen_string_literal: true

require "formula"
require "cli/parser"

module Homebrew
  module_function

  def vendor_gems_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `vendor-gems`

        Install and commit Homebrew's vendored gems.
      EOS
      switch :debug
      max_named 0
    end
  end

  def vendor_gems
    vendor_gems_args.parse

    Homebrew.install_bundler!

    ohai "cd #{HOMEBREW_LIBRARY_PATH}"
    HOMEBREW_LIBRARY_PATH.cd do
      ohai "bundle install --standalone"
      safe_system "bundle", "install", "--standalone"

      ohai "bundle pristine"
      safe_system "bundle", "pristine"

      ohai "git add vendor/bundle"
      system "git", "add", "vendor/bundle"

      if Formula["gpg"].optlinked?
        ENV["PATH"] = PATH.new(ENV["PATH"])
                          .prepend(Formula["gpg"].opt_bin)
      end

      ohai "git commit"
      system "git", "commit", "--message", "brew vendor-gems: commit updates."
    end
  end
end
