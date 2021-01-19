# typed: false
# frozen_string_literal: true

require "formula"
require "cli/parser"

module Homebrew
  extend T::Sig

  module_function

  sig { returns(CLI::Parser) }
  def vendor_gems_args
    Homebrew::CLI::Parser.new do
      description <<~EOS
        Install and commit Homebrew's vendored gems.
      EOS

      comma_array "--update",
                  description: "Update all vendored Gems to the latest version."

      named_args :none
    end
  end

  sig { void }
  def vendor_gems
    args = vendor_gems_args.parse

    Homebrew.install_bundler!

    ohai "cd #{HOMEBREW_LIBRARY_PATH}"
    HOMEBREW_LIBRARY_PATH.cd do
      if args.update
        ohai "bundle update"
        safe_system "bundle", "update", *args.update

        ohai "git add Gemfile.lock"
        system "git", "add", "Gemfile.lock"
      end

      ohai "bundle install --standalone"
      safe_system "bundle", "install", "--standalone"

      ohai "bundle pristine"
      safe_system "bundle", "pristine"

      ohai "git add vendor/bundle"
      system "git", "add", "vendor/bundle"

      Utils::Git.set_name_email!
      Utils::Git.setup_gpg!

      ohai "git commit"
      system "git", "commit", "--message", "brew vendor-gems: commit updates."
    end
  end
end
