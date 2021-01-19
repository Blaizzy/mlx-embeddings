# typed: true
# frozen_string_literal: true

require "formula"
require "cli/parser"

module Homebrew
  extend T::Sig

  module_function

  sig { returns(CLI::Parser) }
  def bump_revision_args
    Homebrew::CLI::Parser.new do
      description <<~EOS
        Create a commit to increment the revision of <formula>. If no revision is
        present, "revision 1" will be added.
      EOS
      switch "-n", "--dry-run",
             description: "Print what would be done rather than doing it."
      flag   "--message=",
             description: "Append <message> to the default commit message."

      named_args :formula, min: 1
    end
  end

  def bump_revision
    args = bump_revision_args.parse

    # As this command is simplifying user-run commands then let's just use a
    # user path, too.
    ENV["PATH"] = ENV["HOMEBREW_PATH"]

    args.named.to_formulae.each do |formula|
      current_revision = formula.revision
      new_revision = current_revision + 1

      if args.dry_run?
        unless args.quiet?
          old_text = "revision #{current_revision}"
          new_text = "revision #{new_revision}"
          if current_revision.zero?
            ohai "add #{new_text.inspect}"
          else
            ohai "replace #{old_text.inspect} with #{new_text.inspect}"
          end
        end
      else
        Homebrew.install_bundler_gems!
        require "utils/ast"

        formula_ast = Utils::AST::FormulaAST.new(formula.path.read)
        if current_revision.zero?
          formula_ast.add_stanza(:revision, new_revision)
        else
          formula_ast.replace_stanza(:revision, new_revision)
        end
        formula.path.atomic_write(formula_ast.process)
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
