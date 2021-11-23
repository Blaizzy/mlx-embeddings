# typed: false
# frozen_string_literal: true

require "cli/parser"

module Homebrew
  extend T::Sig

  module_function

  sig { returns(CLI::Parser) }
  def cat_args
    Homebrew::CLI::Parser.new do
      description <<~EOS
        Display the source of a <formula> or <cask>.
      EOS

      switch "--formula", "--formulae",
             description: "Treat all named arguments as formulae."
      switch "--cask", "--casks",
             description: "Treat all named arguments as casks."

      conflicts "--formula", "--cask"

      named_args [:formula, :cask], number: 1
    end
  end

  def cat
    args = cat_args.parse

    cd HOMEBREW_REPOSITORY
    pager = if Homebrew::EnvConfig.bat?
      require "formula"

      unless Formula["bat"].any_version_installed?
        # The user might want to capture the output of `brew cat ...`
        # Redirect stdout to stderr
        redirect_stdout($stderr) do
          ohai "Installing `bat` for displaying <formula>/<cask> source..."
          safe_system HOMEBREW_BREW_FILE, "install", "bat"
        end
      end
      ENV["BAT_CONFIG_PATH"] = Homebrew::EnvConfig.bat_config_path
      Formula["bat"].opt_bin/"bat"
    else
      "cat"
    end

    safe_system pager, args.named.to_paths.first
  end
end
