# typed: false
# frozen_string_literal: true

require "cxxstdlib"
require "formula"
require "keg"
require "tab"
require "utils/bottles"
require "caveats"
require "cleaner"
require "formula_cellar_checks"
require "install_renamed"
require "debrew"
require "sandbox"
require "development_tools"
require "cache_store"
require "linkage_checker"
require "install"
require "messages"
require "cask/cask_loader"
require "cmd/install"
require "find"
require "utils/spdx"
require "deprecate_disable"
require "unlink"

# Installer for a formula.
#
# @api private
class FormulaInstaller
  extend T::Sig

  include FormulaCellarChecks
  extend Predicable

  attr_reader :formula

  attr_accessor :options, :link_keg

  attr_predicate :installed_as_dependency?, :installed_on_request?
  attr_predicate :show_summary_heading?, :show_header?
  attr_predicate :force_bottle?, :ignore_deps?, :only_deps?, :interactive?, :git?, :force?, :keep_tmp?
  attr_predicate :verbose?, :debug?, :quiet?

  # TODO: Remove when removed from `test-bot`.
  attr_writer :build_bottle

  def initialize(
    formula,
    link_keg: false,
    installed_as_dependency: false,
    installed_on_request: true,
    show_header: false,
    build_bottle: false,
    force_bottle: false,
    bottle_arch: nil,
    ignore_deps: false,
    only_deps: false,
    include_test_formulae: [],
    build_from_source_formulae: [],
    env: nil,
    git: false,
    interactive: false,
    keep_tmp: false,
    cc: nil,
    options: Options.new,
    force: false,
    debug: false,
    quiet: false,
    verbose: false
  )
    @formula = formula
    @env = env
    @force = force
    @keep_tmp = keep_tmp
    @link_keg = !formula.keg_only? || link_keg
    @show_header = show_header
    @ignore_deps = ignore_deps
    @only_deps = only_deps
    @build_from_source_formulae = build_from_source_formulae
    @build_bottle = build_bottle
    @bottle_arch = bottle_arch
    @formula.force_bottle ||= force_bottle
    @force_bottle = @formula.force_bottle
    @include_test_formulae = include_test_formulae
    @interactive = interactive
    @git = git
    @cc = cc
    @verbose = verbose
    @quiet = quiet
    @debug = debug
    @installed_as_dependency = installed_as_dependency
    @installed_on_request = installed_on_request
    @options = options
    @requirement_messages = []
    @poured_bottle = false
    @pour_failed = false
    @start_time = nil
  end

  def self.attempted
    @attempted ||= Set.new
  end

  sig { void }
  def self.clear_attempted
    @attempted = Set.new
  end

  def self.installed
    @installed ||= Set.new
  end

  sig { void }
  def self.clear_installed
    @installed = Set.new
  end

  # When no build tools are available and build flags are passed through ARGV,
  # it's necessary to interrupt the user before any sort of installation
  # can proceed. Only raises when the user has no developer tools.
  def self.prevent_build_flags(args)
    return if DevelopmentTools.installed?

    build_flags = []

    build_flags << "--HEAD" if args.HEAD?
    build_flags << "--universal" if args.universal?
    build_flags << "--build-bottle" if args.build_bottle?
    build_flags << "--build-from-source" if args.build_from_source?

    return if build_flags.empty?

    all_bottled = args.named.to_formulae.all?(&:bottled?)
    raise BuildFlagsError.new(build_flags, bottled: all_bottled)
  end

  sig { returns(T::Boolean) }
  def build_from_source?
    @build_from_source_formulae.include?(formula.full_name)
  end

  sig { returns(T::Boolean) }
  def include_test?
    @include_test_formulae.include?(formula.full_name)
  end

  sig { returns(T::Boolean) }
  def build_bottle?
    return false unless @build_bottle

    !formula.bottle_disabled?
  end

  sig { params(output_warning: T::Boolean).returns(T::Boolean) }
  def pour_bottle?(output_warning: false)
    return false if @pour_failed

    return false if !formula.bottle_tag? && !formula.local_bottle_path
    return true  if force_bottle?
    return false if build_from_source? || build_bottle? || interactive?
    return false if @cc
    return false unless options.empty?
    return false if formula.bottle_disabled?

    unless formula.pour_bottle?
      if output_warning && formula.pour_bottle_check_unsatisfied_reason
        opoo <<~EOS
          Building #{formula.full_name} from source:
            #{formula.pour_bottle_check_unsatisfied_reason}
        EOS
      end
      return false
    end

    bottle = formula.bottle_specification
    unless bottle.compatible_locations?
      if output_warning
        opoo <<~EOS
          Building #{formula.full_name} from source as the bottle needs:
          - HOMEBREW_CELLAR: #{bottle.cellar} (yours is #{HOMEBREW_CELLAR})
          - HOMEBREW_PREFIX: #{bottle.prefix} (yours is #{HOMEBREW_PREFIX})
        EOS
      end
      return false
    end

    true
  end

  sig { params(dep: Formula, build: BuildOptions).returns(T::Boolean) }
  def install_bottle_for?(dep, build)
    return pour_bottle? if dep == formula

    @build_from_source_formulae.exclude?(dep.full_name) &&
      dep.bottle.present? &&
      dep.pour_bottle? &&
      build.used_options.empty? &&
      dep.bottle&.compatible_locations?
  end

  sig { void }
  def prelude
    Tab.clear_cache
    verify_deps_exist unless ignore_deps?
    forbidden_license_check

    check_install_sanity
  end

  sig { void }
  def verify_deps_exist
    begin
      compute_dependencies
    rescue TapFormulaUnavailableError => e
      raise if e.tap.installed?

      e.tap.install
      retry
    end
  rescue FormulaUnavailableError => e
    e.dependent = formula.full_name
    raise
  end

  def check_install_sanity
    raise FormulaInstallationAlreadyAttemptedError, formula if self.class.attempted.include?(formula)

    if force_bottle? && !pour_bottle?
      raise CannotInstallFormulaError, "--force-bottle passed but #{formula.full_name} has no bottle!"
    end

    if Homebrew.default_prefix? && !Homebrew::EnvConfig.developer? &&
       # TODO: re-enable this on Linux when we merge linuxbrew-core into
       # homebrew-core and have full bottle coverage.
       (OS.mac? || ENV["CI"]) &&
       !build_from_source? && !build_bottle? &&
       !installed_as_dependency? &&
       formula.tap&.core_tap? && !formula.bottle_unneeded? && !formula.any_version_installed? &&
       # Integration tests override homebrew-core locations
       ENV["HOMEBREW_TEST_TMPDIR"].nil? &&
       !pour_bottle?
      message = if !formula.pour_bottle? && formula.pour_bottle_check_unsatisfied_reason
        formula_message = formula.pour_bottle_check_unsatisfied_reason
        formula_message[0] = formula_message[0].downcase

        "#{formula}: #{formula_message}"
      else
        <<~EOS
          #{formula}: no bottle available!
        EOS
      end
      message += <<~EOS
        You can try to install from source with:
          brew install --build-from-source #{formula}
        Please note building from source is unsupported. You will encounter build
        failures with some formulae. If you experience any issues please create pull
        requests instead of asking for help on Homebrew's GitHub, Twitter or any other
        official channels.
      EOS
      raise CannotInstallFormulaError, message
    end

    type, reason = DeprecateDisable.deprecate_disable_info formula

    if type.present?
      case type
      when :deprecated
        if reason.present?
          opoo "#{formula.full_name} has been deprecated because it #{reason}!"
        else
          opoo "#{formula.full_name} has been deprecated!"
        end
      when :disabled
        if reason.present?
          raise CannotInstallFormulaError, "#{formula.full_name} has been disabled because it #{reason}!"
        end

        raise CannotInstallFormulaError, "#{formula.full_name} has been disabled!"
      end
    end

    return if ignore_deps?

    recursive_deps = formula.recursive_dependencies
    recursive_formulae = recursive_deps.map(&:to_formula)

    recursive_dependencies = []
    invalid_arch_dependencies = []
    recursive_formulae.each do |dep|
      dep_recursive_dependencies = dep.recursive_dependencies.map(&:to_s)
      if dep_recursive_dependencies.include?(formula.name)
        recursive_dependencies << "#{formula.full_name} depends on #{dep.full_name}"
        recursive_dependencies << "#{dep.full_name} depends on #{formula.full_name}"
      end

      if (tab = Tab.for_formula(dep)) && tab.arch.present? && tab.arch.to_s != Hardware::CPU.arch.to_s
        invalid_arch_dependencies << "#{dep} was built for #{tab.arch}"
      end
    end

    unless recursive_dependencies.empty?
      raise CannotInstallFormulaError, <<~EOS
        #{formula.full_name} contains a recursive dependency on itself:
          #{recursive_dependencies.join("\n  ")}
      EOS
    end

    if recursive_formulae.flat_map(&:recursive_dependencies)
                         .map(&:to_s)
                         .include?(formula.name)
      raise CannotInstallFormulaError, <<~EOS
        #{formula.full_name} contains a recursive dependency on itself!
      EOS
    end

    unless invalid_arch_dependencies.empty?
      raise CannotInstallFormulaError, <<~EOS
        #{formula.full_name} dependencies not built for the #{Hardware::CPU.arch} CPU architecture:
          #{invalid_arch_dependencies.join("\n  ")}
      EOS
    end

    pinned_unsatisfied_deps = recursive_deps.select do |dep|
      dep.to_formula.pinned? && !dep.satisfied?(inherited_options_for(dep))
    end

    return if pinned_unsatisfied_deps.empty?

    raise CannotInstallFormulaError,
          "You must `brew unpin #{pinned_unsatisfied_deps * " "}` as installing " \
          "#{formula.full_name} requires the latest version of pinned dependencies"
  end

  def build_bottle_preinstall
    @etc_var_dirs ||= [HOMEBREW_PREFIX/"etc", HOMEBREW_PREFIX/"var"]
    @etc_var_preinstall = Find.find(*@etc_var_dirs.select(&:directory?)).to_a
  end

  def build_bottle_postinstall
    @etc_var_postinstall = Find.find(*@etc_var_dirs.select(&:directory?)).to_a
    (@etc_var_postinstall - @etc_var_preinstall).each do |file|
      Pathname.new(file).cp_path_sub(HOMEBREW_PREFIX, formula.bottle_prefix)
    end
  end

  sig { void }
  def install
    lock

    start_time = Time.now
    if !formula.bottle_unneeded? && !pour_bottle? && DevelopmentTools.installed?
      Homebrew::Install.perform_build_from_source_checks
    end

    # not in initialize so upgrade can unlink the active keg before calling this
    # function but after instantiating this class so that it can avoid having to
    # relink the active keg if possible (because it is slow).
    if formula.linked_keg.directory?
      message = <<~EOS
        #{formula.name} #{formula.linked_version} is already installed
      EOS
      if formula.outdated? && !formula.head?
        message += <<~EOS
          To upgrade to #{formula.pkg_version}, run:
            brew upgrade #{formula.full_name}
        EOS
      elsif only_deps?
        message = nil
      else
        # some other version is already installed *and* linked
        message += <<~EOS
          To install #{formula.pkg_version}, first run:
            brew unlink #{formula.name}
        EOS
      end
      raise CannotInstallFormulaError, message if message
    end

    # Warn if a more recent version of this formula is available in the tap.
    begin
      if formula.pkg_version < (v = Formulary.factory(formula.full_name, force_bottle: force_bottle?).pkg_version)
        opoo "#{formula.full_name} #{v} is available and more recent than version #{formula.pkg_version}."
      end
    rescue FormulaUnavailableError
      nil
    end

    check_conflicts

    raise UnbottledError, [formula] if !pour_bottle? && !formula.bottle_unneeded? && !DevelopmentTools.installed?

    unless ignore_deps?
      deps = compute_dependencies
      if ((pour_bottle? && !DevelopmentTools.installed?) || build_bottle?) &&
         (unbottled = unbottled_dependencies(deps)).presence
        # Check that each dependency in deps has a bottle available, terminating
        # abnormally with a UnbottledError if one or more don't.
        raise UnbottledError, unbottled
      end

      install_dependencies(deps)
    end

    return if only_deps?

    if build_bottle? && (arch = @bottle_arch) && Hardware::CPU.optimization_flags.exclude?(arch.to_sym)
      raise CannotInstallFormulaError, "Unrecognized architecture for --bottle-arch: #{arch}"
    end

    formula.deprecated_flags.each do |deprecated_option|
      old_flag = deprecated_option.old_flag
      new_flag = deprecated_option.current_flag
      opoo "#{formula.full_name}: #{old_flag} was deprecated; using #{new_flag} instead!"
    end

    options = display_options(formula).join(" ")
    oh1 "Installing #{Formatter.identifier(formula.full_name)} #{options}".strip if show_header?

    unless formula.tap&.private?
      action = "#{formula.full_name} #{options}".strip
      Utils::Analytics.report_event("install", action)

      Utils::Analytics.report_event("install_on_request", action) if installed_on_request?
    end

    self.class.attempted << formula

    if pour_bottle?
      begin
        pour
      rescue Exception => e # rubocop:disable Lint/RescueException
        # any exceptions must leave us with nothing installed
        ignore_interrupts do
          begin
            formula.prefix.rmtree if formula.prefix.directory?
          rescue Errno::EACCES, Errno::ENOTEMPTY
            odie <<~EOS
              Could not remove #{formula.prefix.basename} keg! Do so manually:
                sudo rm -rf #{formula.prefix}
            EOS
          end
          formula.rack.rmdir_if_possible
        end
        raise if Homebrew::EnvConfig.developer? ||
                 Homebrew::EnvConfig.no_bottle_source_fallback? ||
                 force_bottle? ||
                 e.is_a?(Interrupt)

        @pour_failed = true
        onoe e.message
        opoo "Bottle installation failed: building from source."
        raise UnbottledError, [formula] unless DevelopmentTools.installed?

        compute_and_install_dependencies unless ignore_deps?
      else
        @poured_bottle = true
      end
    end

    puts_requirement_messages

    build_bottle_preinstall if build_bottle?

    unless @poured_bottle
      build
      clean

      # Store the formula used to build the keg in the keg.
      formula_contents = if formula.local_bottle_path
        Utils::Bottles.formula_contents formula.local_bottle_path, name: formula.name
      else
        formula.path.read
      end
      s = formula_contents.gsub(/  bottle do.+?end\n\n?/m, "")
      brew_prefix = formula.prefix/".brew"
      brew_prefix.mkdir
      Pathname(brew_prefix/"#{formula.name}.rb").atomic_write(s)

      keg = Keg.new(formula.prefix)
      tab = Tab.for_keg(keg)
      tab.installed_as_dependency = installed_as_dependency?
      tab.installed_on_request = installed_on_request?
      tab.write
    end

    build_bottle_postinstall if build_bottle?

    opoo "Nothing was installed to #{formula.prefix}" unless formula.latest_version_installed?
    end_time = Time.now
    Homebrew.messages.formula_installed(formula, end_time - start_time)
  end

  def check_conflicts
    return if force?

    conflicts = formula.conflicts.select do |c|
      f = Formulary.factory(c.name)
    rescue TapFormulaUnavailableError
      # If the formula name is a fully-qualified name let's silently
      # ignore it as we don't care about things used in taps that aren't
      # currently tapped.
      false
    rescue FormulaUnavailableError => e
      # If the formula name doesn't exist any more then complain but don't
      # stop installation from continuing.
      opoo <<~EOS
        #{formula}: #{e.message}
        'conflicts_with \"#{c.name}\"' should be removed from #{formula.path.basename}.
      EOS

      raise if Homebrew::EnvConfig.developer?

      $stderr.puts "Please report this issue to the #{formula.tap} tap (not Homebrew/brew or Homebrew/core)!"
      false
    else # rubocop:disable Layout/ElseAlignment
      f.linked_keg.exist? && f.opt_prefix.exist?
    end

    raise FormulaConflictError.new(formula, conflicts) unless conflicts.empty?
  end

  # Compute and collect the dependencies needed by the formula currently
  # being installed.
  def compute_dependencies
    req_map, req_deps = expand_requirements
    check_requirements(req_map)
    expand_dependencies(req_deps + formula.deps)
  end

  def unbottled_dependencies(deps)
    deps.map(&:first).map(&:to_formula).reject do |dep_f|
      next false unless dep_f.pour_bottle?

      dep_f.bottle_unneeded? || dep_f.bottled?
    end
  end

  def compute_and_install_dependencies
    deps = compute_dependencies
    install_dependencies(deps)
  end

  def check_requirements(req_map)
    @requirement_messages = []
    fatals = []

    req_map.each_pair do |dependent, reqs|
      reqs.each do |req|
        next if dependent.latest_version_installed? && req.name == "maximummacos"

        @requirement_messages << "#{dependent}: #{req.message}"
        fatals << req if req.fatal?
      end
    end

    return if fatals.empty?

    puts_requirement_messages
    raise UnsatisfiedRequirements, fatals
  end

  def runtime_requirements(formula)
    runtime_deps = formula.runtime_formula_dependencies(undeclared: false)
    recursive_requirements = formula.recursive_requirements do |dependent, _|
      Requirement.prune unless runtime_deps.include?(dependent)
    end
    (recursive_requirements.to_a + formula.requirements.to_a).reject(&:build?).uniq
  end

  def expand_requirements
    unsatisfied_reqs = Hash.new { |h, k| h[k] = [] }
    req_deps = []
    formulae = [formula]
    formula_deps_map = Dependency.expand(formula)
                                 .index_by(&:name)

    while f = formulae.pop
      runtime_requirements = runtime_requirements(f)
      f.recursive_requirements do |dependent, req|
        build = effective_build_options_for(dependent)
        install_bottle_for_dependent = install_bottle_for?(dependent, build)

        keep_build_test = false
        keep_build_test ||= runtime_requirements.include?(req)
        keep_build_test ||= req.test? && include_test? && dependent == f
        keep_build_test ||= req.build? && !install_bottle_for_dependent && !dependent.latest_version_installed?

        if req.prune_from_option?(build) ||
           req.satisfied?(env: @env, cc: @cc, build_bottle: @build_bottle, bottle_arch: @bottle_arch) ||
           ((req.build? || req.test?) && !keep_build_test) ||
           formula_deps_map[dependent.name]&.build?
          Requirement.prune
        else
          unsatisfied_reqs[dependent] << req
        end
      end
    end

    # Merge the repeated dependencies, which may have different tags.
    req_deps = Dependency.merge_repeats(req_deps)

    [unsatisfied_reqs, req_deps]
  end

  def expand_dependencies(deps)
    inherited_options = Hash.new { |hash, key| hash[key] = Options.new }
    pour_bottle = pour_bottle?

    expanded_deps = Dependency.expand(formula, deps) do |dependent, dep|
      inherited_options[dep.name] |= inherited_options_for(dep)
      build = effective_build_options_for(
        dependent,
        inherited_options.fetch(dependent.name, []),
      )

      keep_build_test = false
      keep_build_test ||= dep.test? && include_test? && @include_test_formulae.include?(dependent.full_name)
      keep_build_test ||= dep.build? && !install_bottle_for?(dependent, build) && !dependent.latest_version_installed?

      if dep.prune_from_option?(build) || ((dep.build? || dep.test?) && !keep_build_test)
        Dependency.prune
      elsif dep.satisfied?(inherited_options[dep.name])
        Dependency.skip
      else
        pour_bottle ||= install_bottle_for?(dep.to_formula, build)
      end
    end

    if pour_bottle && !Keg.bottle_dependencies.empty?
      bottle_deps = if Keg.bottle_dependencies.exclude?(formula.name)
        Keg.bottle_dependencies
      elsif Keg.relocation_formulae.exclude?(formula.name)
        Keg.relocation_formulae
      else
        []
      end
      bottle_deps = bottle_deps.map { |formula| Dependency.new(formula) }
                               .reject do |dep|
        inherited_options[dep.name] |= inherited_options_for(dep)
        dep.satisfied? inherited_options[dep.name]
      end
      expanded_deps = Dependency.merge_repeats(bottle_deps + expanded_deps)
    end

    expanded_deps.map { |dep| [dep, inherited_options[dep.name]] }
  end

  def effective_build_options_for(dependent, inherited_options = [])
    args  = dependent.build.used_options
    args |= (dependent == formula) ? options : inherited_options
    args |= Tab.for_formula(dependent).used_options
    args &= dependent.options
    BuildOptions.new(args, dependent.options)
  end

  def display_options(formula)
    options = if formula.head?
      ["--HEAD"]
    else
      []
    end
    options += effective_build_options_for(formula).used_options.to_a
    options
  end

  sig { params(dep: Dependency).returns(Options) }
  def inherited_options_for(dep)
    inherited_options = Options.new
    u = Option.new("universal")
    if (options.include?(u) || formula.require_universal_deps?) && !dep.build? && dep.to_formula.option_defined?(u)
      inherited_options << u
    end
    inherited_options
  end

  sig { params(deps: T::Array[[Formula, Options]]).void }
  def install_dependencies(deps)
    if deps.empty? && only_deps?
      puts "All dependencies for #{formula.full_name} are satisfied."
    elsif !deps.empty?
      oh1 "Installing dependencies for #{formula.full_name}: " \
          "#{deps.map(&:first).map(&Formatter.method(:identifier)).to_sentence}",
          truncate: false
      deps.each { |dep, options| install_dependency(dep, options) }
    end

    @show_header = true unless deps.empty?
  end

  sig { params(dep: Dependency).void }
  def fetch_dependency(dep)
    df = dep.to_formula
    fi = FormulaInstaller.new(
      df,
      force_bottle:               false,
      # When fetching we don't need to recurse the dependency tree as it's already
      # been done for us in `compute_dependencies` and there's no requirement to
      # fetch in a particular order.
      ignore_deps:                true,
      installed_as_dependency:    true,
      include_test_formulae:      @include_test_formulae,
      build_from_source_formulae: @build_from_source_formulae,
      keep_tmp:                   keep_tmp?,
      force:                      force?,
      debug:                      debug?,
      quiet:                      quiet?,
      verbose:                    verbose?,
    )
    fi.fetch
  end

  sig { params(dep: Dependency, inherited_options: Options).void }
  def install_dependency(dep, inherited_options)
    df = dep.to_formula
    tab = Tab.for_formula(df)

    if df.linked_keg.directory?
      linked_keg = Keg.new(df.linked_keg.resolved_path)
      keg_had_linked_keg = true
      keg_was_linked = linked_keg.linked?
      linked_keg.unlink
    end

    if df.latest_version_installed?
      installed_keg = Keg.new(df.prefix)
      tmp_keg = Pathname.new("#{installed_keg}.tmp")
      installed_keg.rename(tmp_keg)
    end

    tab_tap = tab.source["tap"]
    if tab_tap.present? && df.tap.present? && df.tap.to_s != tab_tap.to_s
      odie <<~EOS
        #{df} is already installed from #{tab_tap}!
        Please `brew uninstall #{df}` first."
      EOS
    end

    options = Options.new
    options |= tab.used_options
    options |= Tab.remap_deprecated_options(df.deprecated_options, dep.options)
    options |= inherited_options
    options &= df.options

    fi = FormulaInstaller.new(
      df,
      **{
        options:                    options,
        link_keg:                   keg_had_linked_keg ? keg_was_linked : nil,
        installed_as_dependency:    true,
        installed_on_request:       df.any_version_installed? && tab.installed_on_request,
        force_bottle:               false,
        include_test_formulae:      @include_test_formulae,
        build_from_source_formulae: @build_from_source_formulae,
        keep_tmp:                   keep_tmp?,
        force:                      force?,
        debug:                      debug?,
        quiet:                      quiet?,
        verbose:                    verbose?,
      },
    )
    fi.prelude
    oh1 "Installing #{formula.full_name} dependency: #{Formatter.identifier(dep.name)}"
    fi.install
    fi.finish
  rescue Exception => e # rubocop:disable Lint/RescueException
    ignore_interrupts do
      tmp_keg.rename(installed_keg) if tmp_keg && !installed_keg.directory?
      linked_keg.link(verbose: verbose?) if keg_was_linked
    end
    raise unless e.is_a? FormulaInstallationAlreadyAttemptedError

    # We already attempted to install f as part of another formula's
    # dependency tree. In that case, don't generate an error, just move on.
    nil
  else
    ignore_interrupts { tmp_keg.rmtree if tmp_keg&.directory? }
  end

  sig { void }
  def caveats
    return if only_deps?

    audit_installed if Homebrew::EnvConfig.developer?

    caveats = Caveats.new(formula)

    return if caveats.empty?

    @show_summary_heading = true
    ohai "Caveats", caveats.to_s
    Homebrew.messages.record_caveats(formula, caveats)
  end

  sig { void }
  def finish
    return if only_deps?

    ohai "Finishing up" if verbose?

    install_plist

    keg = Keg.new(formula.prefix)
    link(keg)

    fix_dynamic_linkage(keg) if !@poured_bottle || !formula.bottle_specification.skip_relocation?

    if build_bottle?
      ohai "Not running 'post_install' as we're building a bottle"
      puts "You can run it manually using:"
      puts "  brew postinstall #{formula.full_name}"
    else
      post_install
    end

    # Updates the cache for a particular formula after doing an install
    CacheStoreDatabase.use(:linkage) do |db|
      break unless db.created?

      LinkageChecker.new(keg, formula, cache_db: db, rebuild_cache: true)
    end

    # Update tab with actual runtime dependencies
    tab = Tab.for_keg(keg)
    Tab.clear_cache
    f_runtime_deps = formula.runtime_dependencies(read_from_tab: false)
    tab.runtime_dependencies = Tab.runtime_deps_hash(f_runtime_deps)
    tab.write

    # let's reset Utils::Git.available? if we just installed git
    Utils::Git.clear_available_cache if formula.name == "git"

    # use installed curl when it's needed and available
    if formula.name == "curl" &&
       !DevelopmentTools.curl_handles_most_https_certificates?
      ENV["HOMEBREW_CURL"] = formula.opt_bin/"curl"
    end

    caveats

    ohai "Summary" if verbose? || show_summary_heading?
    puts summary

    self.class.installed << formula
  ensure
    unlock
  end

  sig { returns(String) }
  def summary
    s = +""
    s << "#{Homebrew::EnvConfig.install_badge}  " unless Homebrew::EnvConfig.no_emoji?
    s << "#{formula.prefix.resolved_path}: #{formula.prefix.abv}"
    s << ", built in #{pretty_duration build_time}" if build_time
    s.freeze
  end

  def build_time
    @build_time ||= Time.now - @start_time if @start_time && !interactive?
  end

  sig { returns(T::Array[String]) }
  def sanitized_argv_options
    args = []
    args << "--ignore-dependencies" if ignore_deps?

    if build_bottle?
      args << "--build-bottle"
      args << "--bottle-arch=#{@bottle_arch}" if @bottle_arch
    end

    args << "--git" if git?
    args << "--interactive" if interactive?
    args << "--verbose" if verbose?
    args << "--debug" if debug?
    args << "--cc=#{@cc}" if @cc
    args << "--keep-tmp" if keep_tmp?

    if @env.present?
      args << "--env=#{@env}"
    elsif formula.env.std? || formula.deps.select(&:build?).any? { |d| d.name == "scons" }
      args << "--env=std"
    end

    args << "--HEAD" if formula.head?

    args
  end

  sig { returns(T::Array[String]) }
  def build_argv
    sanitized_argv_options + options.as_flags
  end

  sig { void }
  def build
    FileUtils.rm_rf(formula.logs)

    @start_time = Time.now

    # 1. formulae can modify ENV, so we must ensure that each
    #    installation has a pristine ENV when it starts, forking now is
    #    the easiest way to do this
    args = %W[
      nice #{RUBY_PATH}
      #{ENV["HOMEBREW_RUBY_WARNINGS"]}
      -I #{$LOAD_PATH.join(File::PATH_SEPARATOR)}
      --
      #{HOMEBREW_LIBRARY_PATH}/build.rb
      #{formula.specified_path}
    ].concat(build_argv)

    Utils.safe_fork do
      if Sandbox.available?
        sandbox = Sandbox.new
        formula.logs.mkpath
        sandbox.record_log(formula.logs/"build.sandbox.log")
        sandbox.allow_write_path(ENV["HOME"]) if interactive?
        sandbox.allow_write_temp_and_cache
        sandbox.allow_write_log(formula)
        sandbox.allow_cvs
        sandbox.allow_fossil
        sandbox.allow_write_xcode
        sandbox.allow_write_cellar(formula)
        sandbox.exec(*args)
      else
        exec(*args)
      end
    end

    formula.update_head_version

    raise "Empty installation" if !formula.prefix.directory? || Keg.new(formula.prefix).empty_installation?
  rescue Exception => e # rubocop:disable Lint/RescueException
    if e.is_a? BuildError
      e.formula = formula
      e.options = display_options(formula)
    end

    ignore_interrupts do
      # any exceptions must leave us with nothing installed
      formula.update_head_version
      formula.prefix.rmtree if formula.prefix.directory?
      formula.rack.rmdir_if_possible
    end
    raise e
  end

  sig { params(keg: Keg).void }
  def link(keg)
    Formula.clear_cache

    unless link_keg
      begin
        keg.optlink(verbose: verbose?)
      rescue Keg::LinkError => e
        ofail "Failed to create #{formula.opt_prefix}"
        puts "Things that depend on #{formula.full_name} will probably not build."
        puts e
      end
      return
    end

    cask_installed_with_formula_name = begin
      Cask::CaskLoader.load(formula.name).installed?
    rescue Cask::CaskUnavailableError, Cask::CaskInvalidError
      false
    end

    if cask_installed_with_formula_name
      ohai "#{formula.name} cask is installed, skipping link."
      return
    end

    if keg.linked?
      opoo "This keg was marked linked already, continuing anyway"
      keg.remove_linked_keg_record
    end

    Homebrew::Unlink.unlink_versioned_formulae(formula, verbose: verbose?)

    link_overwrite_backup = {} # Hash: conflict file -> backup file
    backup_dir = HOMEBREW_CACHE/"Backup"

    begin
      keg.link(verbose: verbose?)
    rescue Keg::ConflictError => e
      conflict_file = e.dst
      if formula.link_overwrite?(conflict_file) && !link_overwrite_backup.key?(conflict_file)
        backup_file = backup_dir/conflict_file.relative_path_from(HOMEBREW_PREFIX).to_s
        backup_file.parent.mkpath
        FileUtils.mv conflict_file, backup_file
        link_overwrite_backup[conflict_file] = backup_file
        retry
      end
      ofail "The `brew link` step did not complete successfully"
      puts "The formula built, but is not symlinked into #{HOMEBREW_PREFIX}"
      puts e
      puts
      puts "Possible conflicting files are:"
      keg.link(dry_run: true, overwrite: true, verbose: verbose?)
      @show_summary_heading = true
    rescue Keg::LinkError => e
      ofail "The `brew link` step did not complete successfully"
      puts "The formula built, but is not symlinked into #{HOMEBREW_PREFIX}"
      puts e
      puts
      puts "You can try again using:"
      puts "  brew link #{formula.name}"
      @show_summary_heading = true
    rescue Exception => e # rubocop:disable Lint/RescueException
      ofail "An unexpected error occurred during the `brew link` step"
      puts "The formula built, but is not symlinked into #{HOMEBREW_PREFIX}"
      puts e
      puts e.backtrace if debug?
      @show_summary_heading = true
      ignore_interrupts do
        keg.unlink
        link_overwrite_backup.each do |origin, backup|
          origin.parent.mkpath
          FileUtils.mv backup, origin
        end
      end
      raise
    end

    return if link_overwrite_backup.empty?

    opoo "These files were overwritten during the `brew link` step:"
    puts link_overwrite_backup.keys
    puts
    puts "They have been backed up to: #{backup_dir}"
    @show_summary_heading = true
  end

  sig { void }
  def install_plist
    return unless formula.plist

    formula.plist_path.atomic_write(formula.plist)
    formula.plist_path.chmod 0644
    log = formula.var/"log"
    log.mkpath if formula.plist.include? log.to_s
  rescue Exception => e # rubocop:disable Lint/RescueException
    ofail "Failed to install plist file"
    odebug e, e.backtrace
  end

  sig { params(keg: Keg).void }
  def fix_dynamic_linkage(keg)
    keg.fix_dynamic_linkage
  rescue Exception => e # rubocop:disable Lint/RescueException
    ofail "Failed to fix install linkage"
    puts "The formula built, but you may encounter issues using it or linking other"
    puts "formulae against it."
    odebug e, e.backtrace
    @show_summary_heading = true
  end

  sig { void }
  def clean
    ohai "Cleaning" if verbose?
    Cleaner.new(formula).clean
  rescue Exception => e # rubocop:disable Lint/RescueException
    opoo "The cleaning step did not complete successfully"
    puts "Still, the installation was successful, so we will link it into your prefix."
    odebug e, e.backtrace
    Homebrew.failed = true
    @show_summary_heading = true
  end

  sig { void }
  def post_install
    args = %W[
      nice #{RUBY_PATH}
      #{ENV["HOMEBREW_RUBY_WARNINGS"]}
      -I #{$LOAD_PATH.join(File::PATH_SEPARATOR)}
      --
      #{HOMEBREW_LIBRARY_PATH}/postinstall.rb
      #{formula.path}
    ]

    Utils.safe_fork do
      if Sandbox.available?
        sandbox = Sandbox.new
        formula.logs.mkpath
        sandbox.record_log(formula.logs/"postinstall.sandbox.log")
        sandbox.allow_write_temp_and_cache
        sandbox.allow_write_log(formula)
        sandbox.allow_write_xcode
        sandbox.deny_write_homebrew_repository
        sandbox.allow_write_cellar(formula)
        Keg::KEG_LINK_DIRECTORIES.each do |dir|
          sandbox.allow_write_path "#{HOMEBREW_PREFIX}/#{dir}"
        end
        sandbox.exec(*args)
      else
        exec(*args)
      end
    end
  rescue Exception => e # rubocop:disable Lint/RescueException
    opoo "The post-install step did not complete successfully"
    puts "You can try again using:"
    puts "  brew postinstall #{formula.full_name}"
    odebug e, e.backtrace, always_display: Homebrew::EnvConfig.developer?
    Homebrew.failed = true
    @show_summary_heading = true
  end

  sig { void }
  def fetch_dependencies
    return if ignore_deps?

    deps = compute_dependencies
    return if deps.empty?

    deps.each { |dep, _options| fetch_dependency(dep) }
  end

  sig { void }
  def fetch
    fetch_dependencies

    return if only_deps?

    if pour_bottle?(output_warning: true)
      begin
        downloader.fetch
      rescue Exception => e # rubocop:disable Lint/RescueException
        raise if Homebrew::EnvConfig.developer? ||
                 Homebrew::EnvConfig.no_bottle_source_fallback? ||
                 force_bottle? ||
                 e.is_a?(Interrupt)

        @pour_failed = true
        onoe e.message
        opoo "Bottle installation failed: building from source."
        fetch_dependencies
      end
    end
    return if pour_bottle?

    formula.fetch_patches
    formula.resources.each(&:fetch)
    downloader.fetch
  end

  def downloader
    if (bottle_path = formula.local_bottle_path)
      LocalBottleDownloadStrategy.new(bottle_path)
    elsif pour_bottle?
      formula.bottle
    else
      formula
    end
  end

  sig { void }
  def pour
    HOMEBREW_CELLAR.cd do
      downloader.stage
    end

    keg = Keg.new(formula.prefix)
    tab = Tab.for_keg(keg)
    Tab.clear_cache

    skip_linkage = formula.bottle_specification.skip_relocation?
    keg.replace_placeholders_with_locations tab.changed_files, skip_linkage: skip_linkage

    tab = Tab.for_keg(keg)

    CxxStdlib.check_compatibility(
      formula, formula.recursive_dependencies,
      Keg.new(formula.prefix), tab.compiler
    )

    tab.tap = formula.tap
    tab.poured_from_bottle = true
    tab.time = Time.now.to_i
    tab.head = HOMEBREW_REPOSITORY.git_head
    tab.source["path"] = formula.specified_path.to_s
    tab.installed_as_dependency = installed_as_dependency?
    tab.installed_on_request = installed_on_request?
    tab.aliases = formula.aliases
    tab.write
  end

  sig { params(output: T.nilable(String)).void }
  def problem_if_output(output)
    return unless output

    opoo output
    @show_summary_heading = true
  end

  def audit_installed
    unless formula.keg_only?
      problem_if_output(check_env_path(formula.bin))
      problem_if_output(check_env_path(formula.sbin))
    end
    super
  end

  def self.locked
    @locked ||= []
  end

  private

  attr_predicate :hold_locks?

  sig { void }
  def lock
    return unless self.class.locked.empty?

    unless ignore_deps?
      formula.recursive_dependencies.each do |dep|
        self.class.locked << dep.to_formula
      end
    end
    self.class.locked.unshift(formula)
    self.class.locked.uniq!
    self.class.locked.each(&:lock)
    @hold_locks = true
  end

  sig { void }
  def unlock
    return unless hold_locks?

    self.class.locked.each(&:unlock)
    self.class.locked.clear
    @hold_locks = false
  end

  def puts_requirement_messages
    return unless @requirement_messages
    return if @requirement_messages.empty?

    $stderr.puts @requirement_messages
  end

  sig { void }
  def forbidden_license_check
    forbidden_licenses = Homebrew::EnvConfig.forbidden_licenses.to_s.dup
    SPDX::ALLOWED_LICENSE_SYMBOLS.each do |s|
      pattern = /#{s.to_s.tr("_", " ")}/i
      forbidden_licenses.sub!(pattern, s.to_s)
    end
    forbidden_licenses = forbidden_licenses.split.to_h do |license|
      [license, SPDX.license_version_info(license)]
    end

    return if forbidden_licenses.blank?

    compute_dependencies.each do |dep, _|
      next if @ignore_deps

      dep_f = dep.to_formula
      next unless SPDX.licenses_forbid_installation? dep_f.license, forbidden_licenses

      raise CannotInstallFormulaError, <<~EOS
        The installation of #{formula.name} has a dependency on #{dep.name} where all its licenses are forbidden:
          #{SPDX.license_expression_to_string dep_f.license}.
      EOS
    end
    return if @only_deps

    return unless SPDX.licenses_forbid_installation? formula.license, forbidden_licenses

    raise CannotInstallFormulaError, <<~EOS
      #{formula.name}'s licenses are all forbidden: #{SPDX.license_expression_to_string formula.license}.
    EOS
  end
end
