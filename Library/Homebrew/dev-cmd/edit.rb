# frozen_string_literal: true

require "formula"
require "cli/parser"

module Homebrew
  module_function

  def edit_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `edit` [<formula>]

        Open <formula> in the editor set by `EDITOR` or `HOMEBREW_EDITOR`, or open the
        Homebrew repository for editing if no formula is provided.
      EOS
    end
  end

  def edit
    args = edit_args.parse

    unless (HOMEBREW_REPOSITORY/".git").directory?
      raise <<~EOS
        Changes will be lost!
        The first time you `brew update`, all local changes will be lost; you should
        thus `brew update` before you `brew edit`!
      EOS
    end

    paths = args.named.to_formulae_paths.presence

    # If no brews are listed, open the project root in an editor.
    paths ||= [HOMEBREW_REPOSITORY]

    exec_editor(*paths)
  end
end
