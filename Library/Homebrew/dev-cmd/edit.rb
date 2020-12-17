# typed: strict
# frozen_string_literal: true

require "formula"
require "cli/parser"

module Homebrew
  extend T::Sig

  module_function

  sig { returns(CLI::Parser) }
  def edit_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `edit` [<formula>|<cask>]

        Open a <formula> or <cask> in the editor set by `EDITOR` or `HOMEBREW_EDITOR`,
        or open the Homebrew repository for editing if no formula is provided.
      EOS

      switch "--formula", "--formulae",
             description: "Treat all named arguments as formulae."
      switch "--cask", "--casks",
             description: "Treat all named arguments as casks."
      conflicts "--formula", "--cask"
    end
  end

  sig { void }
  def edit
    args = edit_args.parse

    unless (HOMEBREW_REPOSITORY/".git").directory?
      raise <<~EOS
        Changes will be lost!
        The first time you `brew update`, all local changes will be lost; you should
        thus `brew update` before you `brew edit`!
      EOS
    end

    paths = args.named.to_paths.select do |path|
      next path if path.exist?

      raise UsageError, "#{path} doesn't exist on disk. " \
                        "Run #{Formatter.identifier("brew create --set-name #{path.basename} $URL")} " \
                        "to create a new Formula!"
    end.presence

    # If no brews are listed, open the project root in an editor.
    paths ||= [HOMEBREW_REPOSITORY]

    exec_editor(*paths)
  end
end
