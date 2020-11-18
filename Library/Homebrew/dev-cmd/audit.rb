# typed: false
# frozen_string_literal: true

require "formula"
require "formula_versions"
require "utils/curl"
require "utils/github/actions"
require "utils/shared_audits"
require "utils/spdx"
require "extend/ENV"
require "formula_cellar_checks"
require "cmd/search"
require "style"
require "date"
require "missing_formula"
require "digest"
require "cli/parser"
require "json"
require "formula_auditor"
require "tap_auditor"

module Homebrew
  extend T::Sig

  module_function

  sig { returns(CLI::Parser) }
  def audit_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `audit` [<options>] [<formula>]

        Check <formula> for Homebrew coding style violations. This should be run before
        submitting a new formula. If no <formula> are provided, check all locally
        available formulae and skip style checks. Will exit with a non-zero status if any
        errors are found.
      EOS
      switch "--strict",
             description: "Run additional, stricter style checks."
      switch "--git",
             description: "Run additional, slower style checks that navigate the Git repository."
      switch "--online",
             description: "Run additional, slower style checks that require a network connection."
      switch "--new-formula",
             description: "Run various additional style checks to determine if a new formula is eligible "\
                          "for Homebrew. This should be used when creating new formula and implies "\
                          "`--strict` and `--online`."
      flag   "--tap=",
             description: "Check the formulae within the given tap, specified as <user>`/`<repo>."
      switch "--fix",
             description: "Fix style violations automatically using RuboCop's auto-correct feature."
      switch "--display-cop-names",
             description: "Include the RuboCop cop name for each violation in the output."
      switch "--display-filename",
             description: "Prefix every line of output with the file or formula name being audited, to "\
                          "make output easy to grep."
      switch "--skip-style",
             description: "Skip running non-RuboCop style checks. Useful if you plan on running "\
                          "`brew style` separately. Enabled by default unless a formula is specified by name."
      switch "-D", "--audit-debug",
             description: "Enable debugging and profiling of audit methods."
      comma_array "--only",
                  description: "Specify a comma-separated <method> list to only run the methods named "\
                               "`audit_`<method>."
      comma_array "--except",
                  description: "Specify a comma-separated <method> list to skip running the methods named "\
                               "`audit_`<method>."
      comma_array "--only-cops",
                  description: "Specify a comma-separated <cops> list to check for violations of only the listed "\
                               "RuboCop cops."
      comma_array "--except-cops",
                  description: "Specify a comma-separated <cops> list to skip checking for violations of the listed "\
                               "RuboCop cops."

      conflicts "--only", "--except"
      conflicts "--only-cops", "--except-cops", "--strict"
      conflicts "--only-cops", "--except-cops", "--only"
      conflicts "--display-cop-names", "--skip-style"
      conflicts "--display-cop-names", "--only-cops"
      conflicts "--display-cop-names", "--except-cops"
    end
  end

  def audit
    args = audit_args.parse

    Homebrew.auditing = true
    inject_dump_stats!(FormulaAuditor, /^audit_/) if args.audit_debug?

    formula_count = 0
    problem_count = 0
    corrected_problem_count = 0
    new_formula_problem_count = 0
    new_formula = args.new_formula?
    strict = new_formula || args.strict?
    online = new_formula || args.online?
    git = args.git?
    skip_style = args.skip_style? || args.no_named? || args.tap

    ENV.activate_extensions!
    ENV.setup_build_environment

    audit_formulae = if args.tap
      Tap.fetch(args.tap).formula_names.map { |name| Formula[name] }
    elsif args.no_named?
      Formula
    else
      args.named.to_resolved_formulae
    end
    style_files = args.named.to_formulae_paths unless skip_style

    only_cops = args.only_cops
    except_cops = args.except_cops
    options = { fix: args.fix?, debug: args.debug?, verbose: args.verbose? }

    if only_cops
      options[:only_cops] = only_cops
    elsif args.new_formula?
      nil
    elsif except_cops
      options[:except_cops] = except_cops
    elsif !strict
      options[:except_cops] = [:FormulaAuditStrict]
    end

    # Run tap audits first
    tap_problem_count = 0
    tap_count = 0
    Tap.each do |tap|
      next if args.tap && tap != args.tap

      ta = TapAuditor.new tap, strict: args.strict?
      ta.audit

      next if ta.problems.blank?

      tap_count += 1
      tap_problem_count += ta.problems.size
      tap_problem_lines = format_problem_lines(ta.problems)

      puts "#{tap.name}:", tap_problem_lines.map { |s| "  #{s}" }
    end

    # Check style in a single batch run up front for performance
    style_offenses = Style.check_style_json(style_files, options) if style_files
    # load licenses
    spdx_license_data = SPDX.license_data
    spdx_exception_data = SPDX.exception_data
    new_formula_problem_lines = []
    audit_formulae.sort.each do |f|
      only = only_cops ? ["style"] : args.only
      options = {
        new_formula:          new_formula,
        strict:               strict,
        online:               online,
        git:                  git,
        only:                 only,
        except:               args.except,
        spdx_license_data:    spdx_license_data,
        spdx_exception_data:  spdx_exception_data,
        tap_audit_exceptions: f.tap.audit_exceptions,
      }
      options[:style_offenses] = style_offenses.for_path(f.path) if style_offenses
      options[:display_cop_names] = args.display_cop_names?
      options[:build_stable] = args.build_stable?

      fa = FormulaAuditor.new(f, options)
      fa.audit
      next if fa.problems.empty? && fa.new_formula_problems.empty?

      formula_count += 1
      problem_count += fa.problems.size
      problem_lines = format_problem_lines(fa.problems)
      corrected_problem_count = options[:style_offenses].count(&:corrected?) if options[:style_offenses]
      new_formula_problem_lines = format_problem_lines(fa.new_formula_problems)
      if args.display_filename?
        puts problem_lines.map { |s| "#{f.path}: #{s}" }
      else
        puts "#{f.full_name}:", problem_lines.map { |s| "  #{s}" }
      end

      next unless ENV["GITHUB_ACTIONS"]

      (fa.problems + fa.new_formula_problems).each do |message:, location:|
        annotation = GitHub::Actions::Annotation.new(
          :error, message, file: f.path, line: location&.line, column: location&.column
        )
        puts annotation if annotation.relevant?
      end
    end

    new_formula_problem_count += new_formula_problem_lines.size
    puts new_formula_problem_lines.map { |s| "  #{s}" }

    total_problems_count = problem_count + new_formula_problem_count + tap_problem_count
    return unless total_problems_count.positive?

    problem_plural = "#{total_problems_count} #{"problem".pluralize(total_problems_count)}"
    formula_plural = "#{formula_count} #{"formula".pluralize(formula_count)}"
    tap_plural = "#{tap_count} #{"tap".pluralize(tap_count)}"
    corrected_problem_plural = "#{corrected_problem_count} #{"problem".pluralize(corrected_problem_count)}"
    errors_summary = if tap_count.zero?
      "#{problem_plural} in #{formula_plural} detected"
    elsif formula_count.zero?
      "#{problem_plural} in #{tap_plural} detected"
    else
      "#{problem_plural} in #{formula_plural} and #{tap_plural} detected"
    end
    errors_summary += ", #{corrected_problem_plural} corrected" if corrected_problem_count.positive?

    ofail errors_summary
  end

  def format_problem_lines(problems)
    problems.uniq
            .map { |message:, location:| format_problem(message, location) }
  end

  def format_problem(message, location)
    "* #{location&.to_s&.dup&.concat(": ")}#{message.chomp.gsub("\n", "\n    ")}"
  end
end
