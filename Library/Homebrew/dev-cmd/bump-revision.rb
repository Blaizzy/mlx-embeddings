# typed: true
# frozen_string_literal: true

require "formula"
require "cli/parser"
require "utils/ast"

module Homebrew
  extend T::Sig

  module_function

  sig { returns(CLI::Parser) }
  def bump_revision_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `bump-revision` [<options>] <formula> [<formula> ...]

        Create a commit to increment the revision of <formula>. If no revision is
        present, "revision 1" will be added.
      EOS
      switch "-n", "--dry-run",
             description: "Print what would be done rather than doing it."
      flag   "--message=",
             description: "Append <message> to the default commit message."

      min_named :formula
    end
  end

  def bump_revision
    args = bump_revision_args.parse

    # As this command is simplifying user-run commands then let's just use a
    # user path, too.
    ENV["PATH"] = ENV["HOMEBREW_PATH"]

    args.named.to_formulae.each do |formula|
      current_revision = formula.revision
      text = "revision #{current_revision+1}"

      if args.dry_run?
        unless args.quiet?
          if current_revision.zero?
            ohai "add #{text.inspect}"
          else
            old = "revision #{current_revision}"
            ohai "replace #{old.inspect} with #{text.inspect}"
          end
        end
      else
        Utils::Inreplace.inreplace(formula.path) do |s|
          s = s.inreplace_string
          if current_revision.zero?
            Utils::AST.add_formula_stanza!(s, :revision, text)
          else
            Utils::AST.replace_formula_stanza!(s, :revision, text)
          end
        end
      end

      message = "#{formula.name}: revision bump #{args.message}"
      if args.dry_run?
        ohai "git commit --no-edit --verbose --message=#{message} -- #{formula.path}"
      else
        formula.path.parent.cd do
          safe_system "git", "commit", "--no-edit", "--verbose",
                      "--message=#{message}", "--", formula.path
        end
      end
    end
  end
end
