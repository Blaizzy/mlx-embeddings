# typed: true
# frozen_string_literal: true

require "cask_dependent"

# Helper functions for dependencies.
#
# @api private
module DependenciesHelpers
  def args_includes_ignores(args)
    includes = [:recommended?] # included by default
    includes << :build? if args.include_build?
    includes << :test? if args.include_test?
    includes << :optional? if args.include_optional?

    ignores = []
    ignores << :recommended? if args.skip_recommended?
    ignores << :satisfied? if args.missing?

    [includes, ignores]
  end

  def recursive_includes(klass, root_dependent, includes, ignores)
    raise ArgumentError, "Invalid class argument: #{klass}" if klass != Dependency && klass != Requirement

    cache_key = "recursive_includes_#{includes}_#{ignores}"

    klass.expand(root_dependent, cache_key: cache_key) do |dependent, dep|
      klass.prune if ignores.any? { |ignore| dep.send(ignore) }

      # NOTE: Untagged dependencies are runtime dependencies and are included by default.
      klass.prune if !dep.tags.empty? && includes.none? do |include|
        # Ignore indirect test dependencies
        next if include == :test? && dependent != root_dependent

        dep.send(include)
      end

      # If a tap isn't installed, we can't find the dependencies of one of
      # its formulae, and an exception will be thrown if we try.
      Dependency.keep_but_prune_recursive_deps if klass == Dependency && dep.tap && !dep.tap.installed?
    end
  end

  def reject_ignores(dependables, ignores, includes)
    dependables.reject do |dep|
      next false unless ignores.any? { |ignore| dep.send(ignore) }

      includes.none? { |include| dep.send(include) }
    end
  end

  def dependents(formulae_or_casks)
    formulae_or_casks.map do |formula_or_cask|
      if formula_or_cask.is_a?(Formula)
        formula_or_cask
      else
        CaskDependent.new(formula_or_cask)
      end
    end
  end
  module_function :dependents
end
