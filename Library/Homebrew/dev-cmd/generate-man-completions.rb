# typed: true
# frozen_string_literal: true

require "formula"
require "ostruct"
require "completions"
require "manpages"

module Homebrew
  extend T::Sig

  module_function

  sig { returns(CLI::Parser) }
  def generate_man_completions_args
    Homebrew::CLI::Parser.new do
      description <<~EOS
        Generate Homebrew's manpages and shell completions.
      EOS
      switch "--fail-if-not-changed",
             hidden:      true,
             description: "Return a failing status code if no changes are detected in the manpage outputs. " \
                          "This can be used to notify CI when the manpages are out of date. Additionally, " \
                          "the date used in new manpages will match those in the existing manpages (to allow " \
                          "comparison without factoring in the date)."
      named_args :none
    end
  end

  def generate_man_completions
    args = generate_man_completions_args.parse

    odeprecated "brew generate-man-completions --fail-if-not-changed" if args.fail_if_not_changed?

    Commands.rebuild_internal_commands_completion_list
    Manpages.regenerate_man_pages(quiet: args.quiet?)
    Completions.update_shell_completions!

    diff = system_command "git", args: [
      "-C", HOMEBREW_REPOSITORY, "diff", "--exit-code", "docs/Manpage.md", "manpages", "completions"
    ]
    if diff.status.success?
      ofail "No changes to manpage or completions."
    else
      puts "Manpage and completions updated."
    end
  end
end
