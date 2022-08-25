# typed: true
# frozen_string_literal: true

require "os/linux/glibc"

class DependencyCollector
  extend T::Sig

  undef gcc_dep_if_needed
  undef glibc_dep_if_needed
  undef init_global_dep_tree_if_needed!

  sig { params(related_formula_names: T::Set[String]).returns(T.nilable(Dependency)) }
  def gcc_dep_if_needed(related_formula_names)
    return unless DevelopmentTools.system_gcc_too_old?
    return if related_formula_names.include?(GCC)
    return if global_dep_tree[GCC]&.intersect?(related_formula_names)
    return if global_dep_tree[GLIBC]&.intersect?(related_formula_names) # gcc depends on glibc

    Dependency.new(GCC)
  end

  sig { params(related_formula_names: T::Set[String]).returns(T.nilable(Dependency)) }
  def glibc_dep_if_needed(related_formula_names)
    return unless OS::Linux::Glibc.below_ci_version?
    return if global_dep_tree[GLIBC]&.intersect?(related_formula_names)

    Dependency.new(GLIBC)
  end

  private

  GLIBC = "glibc"
  GCC = CompilerSelector.preferred_gcc.freeze

  # Use class variables to avoid this expensive logic needing to be done more
  # than once.
  # rubocop:disable Style/ClassVars
  @@global_dep_tree = {}

  sig { void }
  def init_global_dep_tree_if_needed!
    return unless DevelopmentTools.build_system_too_old?
    return if @@global_dep_tree.present?

    # Defined in precedence order (gcc depends on glibc).
    global_deps = [GLIBC, GCC].freeze

    @@global_dep_tree = global_deps.to_h { |name| [name, Set.new([name])] }

    global_deps.each do |global_dep_name|
      # This is an arbitrary number picked based on testing the current tree
      # depth and just to ensure that this doesn't loop indefinitely if we
      # introduce a circular dependency by mistake.
      maximum_tree_depth = 10
      current_tree_depth = 0

      deps = Formula[global_dep_name].deps
      while deps.present?
        current_tree_depth += 1
        if current_tree_depth > maximum_tree_depth
          raise "maximum tree depth (#{maximum_tree_depth}) exceeded calculating #{global_dep_name} dependency tree!"
        end

        @@global_dep_tree[global_dep_name].merge(deps.map(&:name))
        deps = deps.flat_map { |dep| dep.to_formula.deps }
      end
    end
  end

  sig { returns(T::Hash[String, T::Set[String]]) }
  def global_dep_tree
    @@global_dep_tree
  end
  # rubocop:enable Style/ClassVars
end
