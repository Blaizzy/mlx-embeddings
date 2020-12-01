# typed: true
# frozen_string_literal: true

require "cask_dependent"

# Helper functions for dependencies.
#
# @api private
module DependenciesHelpers
  def args_includes_ignores(args)
    includes = []
    ignores = []

    if args.include_build?
      includes << "build?"
    else
      ignores << "build?"
    end

    if args.include_test?
      includes << "test?"
    else
      ignores << "test?"
    end

    if args.include_optional?
      includes << "optional?"
    else
      ignores << "optional?"
    end

    ignores << "recommended?" if args.skip_recommended?

    [includes, ignores]
  end

  def recursive_includes(klass, root_dependent, includes, ignores)
    type = if klass == Dependency
      :dependencies
    elsif klass == Requirement
      :requirements
    else
      raise ArgumentError, "Invalid class argument: #{klass}"
    end

    root_dependent.send("recursive_#{type}") do |dependent, dep|
      if dep.recommended?
        klass.prune if ignores.include?("recommended?") || dependent.build.without?(dep)
      elsif dep.optional?
        klass.prune if includes.exclude?("optional?") && !dependent.build.with?(dep)
      elsif dep.build? || dep.test?
        keep = false
        keep ||= dep.test? && includes.include?("test?") && dependent == root_dependent
        keep ||= dep.build? && includes.include?("build?")
        klass.prune unless keep
      end

      # If a tap isn't installed, we can't find the dependencies of one of
      # its formulae, and an exception will be thrown if we try.
      if type == :dependencies &&
         dep.is_a?(TapDependency) &&
         !dep.tap.installed?
        Dependency.keep_but_prune_recursive_deps
      end
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
