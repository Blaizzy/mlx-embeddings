require "formula"
require "ostruct"
require "cli_parser"

module Homebrew
  module_function

  def deps_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `deps` [<options>] <formula>

        Show dependencies for <formula>. When given multiple formula arguments,
        show the intersection of dependencies for <formula>.
      EOS
      switch "--1",
        description: "Only show dependencies one level down, instead of recursing."
      switch "-n",
        description: "Show dependencies in topological order."
      switch "--union",
        description: "Show the union of dependencies for <formula>, instead of the intersection."
      switch "--full-name",
        description: "List dependencies by their full name."
      switch "--installed",
        description: "Only list those dependencies that are currently installed."
      switch "--all",
        description: "List all the dependencies for all available formulae."
      switch "--include-build",
        description: "Show `:build` type dependencies for <formula>."
      switch "--include-optional",
        description: "Show `:optional` dependencies for <formula>."
      switch "--include-test",
        description: "Show `:test` dependencies for <formula> (non-recursive)."
      switch "--skip-recommended",
        description: "Skip `:recommended` type dependencies for <formula>."
      switch "--include-requirements",
        description: "Include requirements in addition to dependencies for <formula>."
      switch "--tree",
        description: "Show dependencies as a tree. When given multiple formula arguments "\
                     "output individual trees for every formula."
      switch "--for-each",
        description: "Switch into the mode used by `deps --all`, but only list dependencies "\
                     "for specified formula one specified formula per line. This is used for "\
                     "debugging the `--installed`/`--all` display mode."
      switch :verbose
      switch :debug
      conflicts "--installed", "--all"
      formula_options
    end
  end

  def deps
    deps_args.parse
    mode = OpenStruct.new(
      installed?:  args.installed?,
      tree?:       args.tree?,
      all?:        args.all?,
      topo_order?: args.n?,
      union?:      args.union?,
      for_each?:   args.for_each?,
    )

    if mode.tree?
      if mode.installed?
        puts_deps_tree Formula.installed.sort, !args.send("1?")
      else
        raise FormulaUnspecifiedError if args.remaining.empty?

        puts_deps_tree ARGV.formulae, !args.send("1?")
      end
      return
    elsif mode.all?
      puts_deps Formula.sort
      return
    elsif !args.remaining.empty? && mode.for_each?
      puts_deps ARGV.formulae
      return
    end

    @only_installed_arg = args.installed? &&
                          !args.include_build? &&
                          !args.include_test? &&
                          !args.include_optional? &&
                          !args.skip_recommended?

    if args.remaining.empty?
      raise FormulaUnspecifiedError unless mode.installed?

      puts_deps Formula.installed.sort
      return
    end

    all_deps = deps_for_formulae(ARGV.formulae, !args.send("1?"), &(mode.union? ? :| : :&))
    all_deps = condense_requirements(all_deps)
    all_deps.select!(&:installed?) if mode.installed?
    all_deps.map!(&method(:dep_display_name))
    all_deps.uniq!
    all_deps.sort! unless mode.topo_order?
    puts all_deps
  end

  def condense_requirements(deps)
    return deps if args.include_requirements?

    deps.select { |dep| dep.is_a? Dependency }
  end

  def dep_display_name(dep)
    str = if dep.is_a? Requirement
      if args.include_requirements?
        ":#{dep.display_s}"
      else
        # This shouldn't happen, but we'll put something here to help debugging
        "::#{dep.name}"
      end
    elsif args.full_name?
      dep.to_formula.full_name
    else
      dep.name
    end

    if args.annotate?
      str = "#{str}  [build]" if dep.build?
      str = "#{str}  [test]" if dep.test?
      str = "#{str}  [optional]" if dep.optional?
      str = "#{str}  [recommended]" if dep.recommended?
    end

    str
  end

  def deps_for_formula(f, recursive = false)
    includes, ignores = argv_includes_ignores(ARGV)

    deps = f.runtime_dependencies if @only_installed_arg

    if recursive
      deps ||= recursive_includes(Dependency,  f, includes, ignores)
      reqs   = recursive_includes(Requirement, f, includes, ignores)
    else
      deps ||= reject_ignores(f.deps, ignores, includes)
      reqs   = reject_ignores(f.requirements, ignores, includes)
    end

    deps + reqs.to_a
  end

  def deps_for_formulae(formulae, recursive = false, &block)
    formulae.map { |f| deps_for_formula(f, recursive) }.reduce(&block)
  end

  def puts_deps(formulae)
    formulae.each do |f|
      deps = deps_for_formula(f)
      deps = condense_requirements(deps)
      deps.sort_by!(&:name)
      deps.map!(&method(:dep_display_name))
      puts "#{f.full_name}: #{deps.join(" ")}"
    end
  end

  def puts_deps_tree(formulae, recursive = false)
    formulae.each do |f|
      puts f.full_name
      @dep_stack = []
      recursive_deps_tree(f, "", recursive)
      puts
    end
  end

  def recursive_deps_tree(f, prefix, recursive)
    reqs = f.requirements
    deps = f.deps
    dependables = reqs + deps
    dependables.reject!(&:optional?) unless args.include_optional?
    dependables.reject!(&:build?) unless args.include_build?
    dependables.reject!(&:test?) unless args.include_test?
    dependables.reject!(&:recommended?) if args.skip_recommended?
    max = dependables.length - 1
    @dep_stack.push f.name
    dependables.each_with_index do |dep, i|
      next if !args.include_requirements? && dep.is_a?(Requirement)

      tree_lines = if i == max
        "└──"
      else
        "├──"
      end

      display_s = "#{tree_lines} #{dep_display_name(dep)}"
      is_circular = @dep_stack.include?(dep.name)
      display_s = "#{display_s} (CIRCULAR DEPENDENCY)" if is_circular
      puts "#{prefix}#{display_s}"

      next if !recursive || is_circular

      prefix_addition = if i == max
        "    "
      else
        "│   "
      end

      recursive_deps_tree(Formulary.factory(dep.name), prefix + prefix_addition, true) if dep.is_a? Dependency
    end

    @dep_stack.pop
  end
end
