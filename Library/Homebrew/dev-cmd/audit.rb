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
      description <<~EOS
        Check <formula> for Homebrew coding style violations. This should be run before
        submitting a new formula or cask. If no <formula>|<cask> are provided, check all
        locally available formulae and casks and skip style checks. Will exit with a
        non-zero status if any errors are found.
      EOS
      switch "--strict",
             description: "Run additional, stricter style checks."
      switch "--git",
             description: "Run additional, slower style checks that navigate the Git repository."
      switch "--online",
             description: "Run additional, slower style checks that require a network connection."
      switch "--installed",
             description: "Only check formulae and casks that are currently installed."
      switch "--eval-all",
             description: "Evaluate all available formulae and casks, whether installed or not, to audit them. " \
                          "Implied if HOMEBREW_EVAL_ALL is set."
      switch "--all",
             hidden: true
      switch "--new", "--new-formula", "--new-cask",
             description: "Run various additional style checks to determine if a new formula or cask is eligible " \
                          "for Homebrew. This should be used when creating new formula and implies " \
                          "`--strict` and `--online`."
      switch "--[no-]appcast",
             description: "Audit the appcast."
      switch "--token-conflicts",
             description: "Audit for token conflicts."
      flag   "--tap=",
             description: "Check the formulae within the given tap, specified as <user>`/`<repo>."
      switch "--fix",
             description: "Fix style violations automatically using RuboCop's auto-correct feature."
      switch "--display-cop-names",
             description: "Include the RuboCop cop name for each violation in the output."
      switch "--display-filename",
             description: "Prefix every line of output with the file or formula name being audited, to " \
                          "make output easy to grep."
      switch "--display-failures-only",
             description: "Only display casks that fail the audit. This is the default for formulae."
      switch "--skip-style",
             description: "Skip running non-RuboCop style checks. Useful if you plan on running " \
                          "`brew style` separately. Enabled by default unless a formula is specified by name."
      switch "-D", "--audit-debug",
             description: "Enable debugging and profiling of audit methods."
      comma_array "--only",
                  description: "Specify a comma-separated <method> list to only run the methods named " \
                               "`audit_`<method>."
      comma_array "--except",
                  description: "Specify a comma-separated <method> list to skip running the methods named " \
                               "`audit_`<method>."
      comma_array "--only-cops",
                  description: "Specify a comma-separated <cops> list to check for violations of only the listed " \
                               "RuboCop cops."
      comma_array "--except-cops",
                  description: "Specify a comma-separated <cops> list to skip checking for violations of the " \
                               "listed RuboCop cops."
      switch "--formula", "--formulae",
             description: "Treat all named arguments as formulae."
      switch "--cask", "--casks",
             description: "Treat all named arguments as casks."

      conflicts "--only", "--except"
      conflicts "--only-cops", "--except-cops", "--strict"
      conflicts "--only-cops", "--except-cops", "--only"
      conflicts "--display-cop-names", "--skip-style"
      conflicts "--display-cop-names", "--only-cops"
      conflicts "--display-cop-names", "--except-cops"
      conflicts "--formula", "--cask"
      conflicts "--installed", "--all"

      named_args [:formula, :cask]
    end
  end

  sig { void }
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
    skip_style = args.skip_style? || args.no_named? || args.tap
    no_named_args = false

    ENV.activate_extensions!
    ENV.setup_build_environment

    audit_formulae, audit_casks = if args.tap
      Tap.fetch(args.tap).then do |tap|
        [
          tap.formula_names.map { |name| Formula[name] },
          tap.cask_files.map { |path| Cask::CaskLoader.load(path) },
        ]
      end
    elsif args.installed?
      no_named_args = true
      [Formula.installed, Cask::Caskroom.casks]
    elsif args.no_named?
      if !args.eval_all? && !Homebrew::EnvConfig.eval_all?
        odeprecated "brew audit",
                    "brew audit --eval-all or HOMEBREW_EVAL_ALL"
      end
      no_named_args = true
      [Formula.all, Cask::Cask.all]
    else
      args.named.to_formulae_and_casks
          .partition { |formula_or_cask| formula_or_cask.is_a?(Formula) }
    end
    style_files = args.named.to_paths unless skip_style

    only_cops = args.only_cops
    except_cops = args.except_cops
    style_options = { fix: args.fix?, debug: args.debug?, verbose: args.verbose? }

    if only_cops
      style_options[:only_cops] = only_cops
    elsif args.new_formula?
      nil
    elsif except_cops
      style_options[:except_cops] = except_cops
    elsif !strict
      style_options[:except_cops] = [:FormulaAuditStrict]
    end

    # Run tap audits first
    tap_problem_count = 0
    tap_count = 0
    Tap.each do |tap|
      next if args.tap && tap != args.tap

      ta = TapAuditor.new(tap, strict: args.strict?)
      ta.audit

      next if ta.problems.blank?

      tap_count += 1
      tap_problem_count += ta.problems.size
      tap_problem_lines = format_problem_lines(ta.problems)

      puts "#{tap.name}:", tap_problem_lines.map { |s| "  #{s}" }
    end

    # Check style in a single batch run up front for performance
    style_offenses = Style.check_style_json(style_files, style_options) if style_files
    # load licenses
    spdx_license_data = SPDX.license_data
    spdx_exception_data = SPDX.exception_data
    new_formula_problem_lines = []
    formula_results = audit_formulae.sort.to_h do |f|
      only = only_cops ? ["style"] : args.only
      options = {
        new_formula:         new_formula,
        strict:              strict,
        online:              online,
        git:                 args.git?,
        only:                only,
        except:              args.except,
        spdx_license_data:   spdx_license_data,
        spdx_exception_data: spdx_exception_data,
        style_offenses:      style_offenses ? style_offenses.for_path(f.path) : nil,
        display_cop_names:   args.display_cop_names?,
      }.compact

      fa = FormulaAuditor.new(f, **options)
      fa.audit

      if fa.problems.any? || fa.new_formula_problems.any?
        formula_count += 1
        problem_count += fa.problems.size
        problem_lines = format_problem_lines(fa.problems)
        corrected_problem_count += options.fetch(:style_offenses, []).count(&:corrected?)
        new_formula_problem_lines += format_problem_lines(fa.new_formula_problems)
        if args.display_filename?
          puts problem_lines.map { |s| "#{f.path}: #{s}" }
        else
          puts "#{f.full_name}:", problem_lines.map { |s| "  #{s}" }
        end
      end

      [f.path, { errors: fa.problems + fa.new_formula_problems, warnings: [] }]
    end

    cask_results = if audit_casks.empty?
      {}
    else
      require "cask/cmd/abstract_command"
      require "cask/cmd/audit"

      # For switches, we add `|| nil` so that `nil` will be passed instead of `false` if they aren't set.
      # This way, we can distinguish between "not set" and "set to false".
      Cask::Cmd::Audit.audit_casks(
        *audit_casks,
        download:              nil,
        # No need for `|| nil` for `--[no-]appcast` because boolean switches are already `nil` if not passed
        appcast:               args.appcast?,
        online:                args.online? || nil,
        strict:                args.strict? || nil,
        new_cask:              args.new_cask? || nil,
        token_conflicts:       args.token_conflicts? || nil,
        quarantine:            nil,
        any_named_args:        !no_named_args,
        language:              nil,
        display_passes:        args.verbose? || args.named.count == 1,
        display_failures_only: args.display_failures_only?,
      )
    end

    failed_casks = cask_results.reject { |_, result| result[:errors].empty? }

    cask_count = failed_casks.count

    cask_problem_count = failed_casks.sum { |_, result| result[:warnings].count + result[:errors].count }
    new_formula_problem_count += new_formula_problem_lines.count
    total_problems_count = problem_count + new_formula_problem_count + cask_problem_count + tap_problem_count

    if total_problems_count.positive?
      puts new_formula_problem_lines.map { |s| "  #{s}" }

      errors_summary = "#{total_problems_count} #{"problem".pluralize(total_problems_count)}"

      error_sources = []
      error_sources << "#{formula_count} #{"formula".pluralize(formula_count)}" if formula_count.positive?
      error_sources << "#{cask_count} #{"cask".pluralize(cask_count)}" if cask_count.positive?
      error_sources << "#{tap_count} #{"tap".pluralize(tap_count)}" if tap_count.positive?

      errors_summary += " in #{error_sources.to_sentence}" if error_sources.any?

      errors_summary += " detected"

      if corrected_problem_count.positive?
        errors_summary += ", #{corrected_problem_count} #{"problem".pluralize(corrected_problem_count)} corrected"
      end

      ofail errors_summary
    end

    return unless ENV["GITHUB_ACTIONS"]

    annotations = formula_results.merge(cask_results).flat_map do |path, result|
      (
        result[:warnings].map { |w| [:warning, w] } +
        result[:errors].map { |e| [:error, e] }
      ).map do |type, problem|
        GitHub::Actions::Annotation.new(
          type,
          problem[:message],
          file:   path,
          line:   problem[:location]&.line,
          column: problem[:location]&.column,
        )
      end
    end

    annotations.each do |annotation|
      puts annotation if annotation.relevant?
    end
  end

  def format_problem_lines(problems)
    problems.uniq
            .map { |message:, location:| format_problem(message, location) }
  end

  def format_problem(message, location)
    "* #{location&.to_s&.dup&.concat(": ")}#{message.chomp.gsub("\n", "\n    ")}"
  end
end
