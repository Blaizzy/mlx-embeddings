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
    # gcc is required for libgcc_s.so.1 if glibc or gcc are too old
    return unless DevelopmentTools.build_system_too_old?
    return if building_global_dep_tree?
    return if related_formula_names.include?(GCC)
    return if global_dep_tree[GCC]&.intersect?(related_formula_names)

    Dependency.new(GCC)
  end

  sig { params(related_formula_names: T::Set[String]).returns(T.nilable(Dependency)) }
  def glibc_dep_if_needed(related_formula_names)
    return unless OS::Linux::Glibc.below_ci_version?
    return if building_global_dep_tree?
    return if related_formula_names.include?(GLIBC)
    return if global_dep_tree[GLIBC]&.intersect?(related_formula_names)

    Dependency.new(GLIBC)
  end

  private

  GLIBC = "glibc"
  GCC = CompilerSelector.preferred_gcc.freeze

  sig { void }
  def init_global_dep_tree_if_needed!
    return unless DevelopmentTools.build_system_too_old?
    return if building_global_dep_tree?
    return unless global_dep_tree.empty?

    building_global_dep_tree!
    global_dep_tree[GLIBC] = Set.new(global_deps_for(GLIBC))
    # gcc depends on glibc
    global_dep_tree[GCC] = Set.new([*global_deps_for(GCC), GLIBC, *@@global_dep_tree[GLIBC]])
    built_global_dep_tree!
  end

  sig { params(name: String).returns(T::Array[String]) }
  def global_deps_for(name)
    @global_deps_for ||= {}
    # Always strip out glibc and gcc from all parts of dependency tree when
    # we're calculating their dependency trees. Other parts of Homebrew will
    # catch any circular dependencies.
    @global_deps_for[name] ||= Formula[name].deps.map(&:name).flat_map do |dep|
      [dep, *global_deps_for(dep)].compact
    end.uniq
  end

  # Use class variables to avoid this expensive logic needing to be done more
  # than once.
  # rubocop:disable Style/ClassVars
  @@global_dep_tree = {}
  @@building_global_dep_tree = false

  sig { returns(T::Hash[String, T::Set[String]]) }
  def global_dep_tree
    @@global_dep_tree
  end

  sig { void }
  def building_global_dep_tree!
    @@building_global_dep_tree = true
  end

  sig { void }
  def built_global_dep_tree!
    @@building_global_dep_tree = false
  end

  sig { returns(T::Boolean) }
  def building_global_dep_tree?
    @@building_global_dep_tree.present?
  end
  # rubocop:enable Style/ClassVars
end
