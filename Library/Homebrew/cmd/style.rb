# frozen_string_literal: true

require "json"
require "open3"
require "style"
require "cli/parser"

module Homebrew
  module_function

  def style_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `style` [<options>] [<file>|<tap>|<formula>]

        Check formulae or files for conformance to Homebrew style guidelines.

        Lists of <file>, <tap> and <formula> may not be combined. If none are
        provided, `style` will run style checks on the whole Homebrew library,
        including core code and all formulae.
      EOS
      switch "--fix",
             description: "Fix style violations automatically using RuboCop's auto-correct feature."
      switch "--display-cop-names",
             description: "Include the RuboCop cop name for each violation in the output."
      comma_array "--only-cops",
                  description: "Specify a comma-separated <cops> list to check for violations of only the "\
                               "listed RuboCop cops."
      comma_array "--except-cops",
                  description: "Specify a comma-separated <cops> list to skip checking for violations of the "\
                               "listed RuboCop cops."
      switch :verbose
      switch :debug
      conflicts "--only-cops", "--except-cops"
    end
  end

  def style
    style_args.parse

    target = if Homebrew.args.named.blank?
      nil
    elsif Homebrew.args.named.any? { |file| File.exist? file }
      Homebrew.args.named
    elsif Homebrew.args.named.any? { |tap| tap.count("/") == 1 }
      Homebrew.args.named.map { |tap| Tap.fetch(tap).path }
    else
      Homebrew.args.formulae.map(&:path)
    end

    only_cops = args.only_cops
    except_cops = args.except_cops

    options = { fix: args.fix? }
    if only_cops
      options[:only_cops] = only_cops
    elsif except_cops
      options[:except_cops] = except_cops
    elsif only_cops.nil? && except_cops.nil?
      options[:except_cops] = %w[FormulaAudit
                                 FormulaAuditStrict
                                 NewFormulaAudit]
    end

    Homebrew.failed = !Style.check_style_and_print(target, options)
  end
end
