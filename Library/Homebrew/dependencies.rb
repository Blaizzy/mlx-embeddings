# frozen_string_literal: true

require "delegate"

class Dependencies < DelegateClass(Array)
  def initialize(*args)
    super(args)
  end

  alias eql? ==

  def optional
    select(&:optional?)
  end

  def recommended
    select(&:recommended?)
  end

  def build
    select(&:build?)
  end

  def required
    select(&:required?)
  end

  def default
    build + required + recommended
  end

  def inspect
    "#<#{self.class.name}: #{to_a}>"
  end
end

class Requirements < DelegateClass(Set)
  def initialize(*args)
    super(Set.new(args))
  end

  def <<(other)
    if other.is_a?(Comparable)
      grep(other.class) do |req|
        return self if req > other

        delete(req)
      end
    end
    super
    self
  end

  def inspect
    "#<#{self.class.name}: {#{to_a.join(", ")}}>"
  end
end

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
        klass.prune if !includes.include?("optional?") && !dependent.build.with?(dep)
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
end
