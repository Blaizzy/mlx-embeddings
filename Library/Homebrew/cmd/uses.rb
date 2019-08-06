# frozen_string_literal: true

require "formula"
require "cli/parser"

# `brew uses foo bar` returns formulae that use both foo and bar
# If you want the union, run the command twice and concatenate the results.
# The intersection is harder to achieve with shell tools.

module Homebrew
  module_function

  def uses_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `uses` [<options>] <formula>

        Show the formulae that specify <formula> as a dependency. When given
        multiple formula arguments, show the intersection of formulae that use
        <formula>.

        By default, `uses` shows all formulae that specify <formula> as a required
        or recommended dependency.

        By default, `uses` shows usage of <formula> by stable builds.
      EOS
      switch "--recursive",
             description: "Resolve more than one level of dependencies."
      switch "--installed",
             description: "Only list installed formulae."
      switch "--include-build",
             description: "Include all formulae that specify <formula> as `:build` type dependency."
      switch "--include-test",
             description: "Include all formulae that specify <formula> as `:test` type dependency."
      switch "--include-optional",
             description: "Include all formulae that specify <formula> as `:optional` type dependency."
      switch "--skip-recommended",
             description: "Skip all formulae that specify <formula> as `:recommended` type dependency."
      switch "--devel",
             description: "Show usage of <formula> by development builds."
      switch "--HEAD",
             description: "Show usage of <formula> by HEAD builds."
      switch :debug
      conflicts "--devel", "--HEAD"
    end
  end

  def uses
    uses_args.parse

    raise FormulaUnspecifiedError if args.remaining.empty?

    used_formulae_missing = false
    used_formulae = begin
      ARGV.formulae
    rescue FormulaUnavailableError => e
      opoo e
      used_formulae_missing = true
      # If the formula doesn't exist: fake the needed formula object name.
      ARGV.named.map { |name| OpenStruct.new name: name, full_name: name }
    end

    formulae = args.installed? ? Formula.installed : Formula
    recursive = args.recursive?
    only_installed_arg = args.installed? &&
                         !args.include_build? &&
                         !args.include_test? &&
                         !args.include_optional? &&
                         !args.skip_recommended?

    includes, ignores = argv_includes_ignores(ARGV)

    uses = formulae.select do |f|
      used_formulae.all? do |ff|
        begin
          deps = f.runtime_dependencies if only_installed_arg
          deps ||= if recursive
            recursive_includes(Dependency, f, includes, ignores)
          else
            reject_ignores(f.deps, ignores, includes)
          end

          deps.any? do |dep|
            begin
              dep.to_formula.full_name == ff.full_name
            rescue
              dep.name == ff.name
            end
          end
        rescue FormulaUnavailableError
          # Silently ignore this case as we don't care about things used in
          # taps that aren't currently tapped.
          next
        end
      end
    end

    return if uses.empty?

    puts Formatter.columns(uses.map(&:full_name).sort)
    odie "Missing formulae should not have dependents!" if used_formulae_missing
  end
end
