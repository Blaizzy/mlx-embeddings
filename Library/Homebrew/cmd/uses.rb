# frozen_string_literal: true

# `brew uses foo bar` returns formulae that use both foo and bar
# If you want the union, run the command twice and concatenate the results.
# The intersection is harder to achieve with shell tools.

require "formula"
require "cli/parser"
require "cask/caskroom"
require "dependencies_helpers"

module Homebrew
  extend DependenciesHelpers

  module_function

  def uses_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `uses` [<options>] <formula>

        Show formulae that specify <formula> as a dependency (i.e. show dependents
        of <formula>). When given multiple formula arguments, show the intersection
        of formulae that use <formula>. By default, `uses` shows all formulae that
        specify <formula> as a required or recommended dependency for their stable builds.
      EOS
      switch "--recursive",
             description: "Resolve more than one level of dependencies."
      switch "--installed",
             description: "Only list formulae that are currently installed."
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

      conflicts "--devel", "--HEAD"
      min_named :formula
    end
  end

  def uses
    args = uses_args.parse

    odeprecated "brew uses --devel" if args.devel?
    odeprecated "brew uses --HEAD" if args.HEAD?

    Formulary.enable_factory_cache!

    used_formulae_missing = false
    used_formulae = begin
      args.formulae
    rescue FormulaUnavailableError => e
      opoo e
      used_formulae_missing = true
      # If the formula doesn't exist: fake the needed formula object name.
      args.named.map { |name| OpenStruct.new name: name, full_name: name }
    end

    use_runtime_dependents = args.installed? &&
                             !used_formulae_missing &&
                             !args.include_build? &&
                             !args.include_test? &&
                             !args.include_optional? &&
                             !args.skip_recommended?

    uses = intersection_of_dependents(use_runtime_dependents, used_formulae, args: args)

    return if uses.empty?

    puts Formatter.columns(uses.map(&:full_name).sort)
    odie "Missing formulae should not have dependents!" if used_formulae_missing
  end

  def intersection_of_dependents(use_runtime_dependents, used_formulae, args:)
    recursive = args.recursive?
    includes, ignores = args_includes_ignores(args)

    if use_runtime_dependents
      used_formulae.map(&:runtime_installed_formula_dependents)
                   .reduce(&:&)
                   .select(&:any_version_installed?) +
        select_used_dependents(dependents(Cask::Caskroom.casks), used_formulae, recursive, includes, ignores)
    else
      deps = if args.installed?
        dependents(Formula.installed + Cask::Caskroom.casks)
      else
        dependents(Formula.to_a + Cask::Cask.to_a)
      end

      select_used_dependents(deps, used_formulae, recursive, includes, ignores)
    end
  end

  def select_used_dependents(dependents, used_formulae, recursive, includes, ignores)
    dependents.select do |d|
      deps = if recursive
        recursive_includes(Dependency, d, includes, ignores)
      else
        reject_ignores(d.deps, ignores, includes)
      end

      used_formulae.all? do |ff|
        deps.any? do |dep|
          match = begin
            dep.to_formula.full_name == ff.full_name if dep.name.include?("/")
          rescue
            nil
          end
          next match unless match.nil?

          dep.name == ff.name
        end
      rescue FormulaUnavailableError
        # Silently ignore this case as we don't care about things used in
        # taps that aren't currently tapped.
        next
      end
    end
  end
end
