# typed: true
# frozen_string_literal: true

require "cli/parser"

module Homebrew
  module_function

  sig { returns(CLI::Parser) }
  def vendor_gems_args
    Homebrew::CLI::Parser.new do
      description <<~EOS
        Install and commit Homebrew's vendored gems.
      EOS

      comma_array "--update",
                  description: "Update the specified list of vendored gems to the latest version."
      switch      "--no-commit",
                  description: "Do not generate a new commit upon completion."
      switch     "--non-bundler-gems",
                 description: "Update vendored gems that aren't using Bundler.",
                 hidden:      true

      named_args :none
    end
  end

  sig { void }
  def vendor_gems
    args = vendor_gems_args.parse

    Homebrew.install_bundler!

    ENV["BUNDLE_WITH"] = Homebrew.valid_gem_groups.join(":")

    # System Ruby does not pick up the correct SDK by default.
    ENV["SDKROOT"] = MacOS.sdk_path if ENV["HOMEBREW_MACOS_SYSTEM_RUBY_NEW_ENOUGH"]

    ohai "cd #{HOMEBREW_LIBRARY_PATH}"
    HOMEBREW_LIBRARY_PATH.cd do
      if args.update
        ohai "bundle update"
        safe_system "bundle", "update", *args.update

        unless args.no_commit?
          ohai "git add Gemfile.lock"
          system "git", "add", "Gemfile.lock"
        end
      end

      ohai "bundle install --standalone"
      safe_system "bundle", "install", "--standalone"

      ohai "bundle pristine"
      safe_system "bundle", "pristine"

      if args.non_bundler_gems?
        %w[
          mechanize
        ].each do |gem|
          ohai "gem install #{gem}"
          safe_system "gem", "install", "mechanize", "--install-dir", "vendor",
                      "--no-document", "--no-wrappers", "--ignore-dependencies", "--force"
          (HOMEBREW_LIBRARY_PATH/"vendor/gems").cd do
            if (source = Pathname.glob("#{gem}-*/").first)
              FileUtils.ln_sf source, gem
            end
          end
        end
      end

      unless args.no_commit?
        ohai "git add vendor"
        system "git", "add", "vendor"

        Utils::Git.set_name_email!
        Utils::Git.setup_gpg!

        ohai "git commit"
        system "git", "commit", "--message", "brew vendor-gems: commit updates."
      end
    end
  end
end
