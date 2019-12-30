# frozen_string_literal: true

require "missing_formula"
require "formula_installer"
require "development_tools"
require "install"
require "search"
require "cleanup"
require "cli/parser"

module Homebrew
  module_function

  extend Search

  def install_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `install` [<options>] <formula>

        Install <formula>. Additional options specific to <formula> may be appended to the command.

        Unless `HOMEBREW_NO_INSTALL_CLEANUP` is set, `brew cleanup` will then be run for the
        installed formulae or, every 30 days, for all formulae.
      EOS
      switch :debug,
             description: "If brewing fails, open an interactive debugging session with access to IRB "\
                          "or a shell inside the temporary build directory."
      flag   "--env=",
             description: "If `std` is passed, use the standard build environment instead of superenv. "\
                          "If `super` is passed, use superenv even if the formula specifies the "\
                          "standard build environment."
      switch "--ignore-dependencies",
             description: "An unsupported Homebrew development flag to skip installing any dependencies of "\
                          "any kind. If the dependencies are not already present, the formula will have issues. "\
                          "If you're not developing Homebrew, consider adjusting your PATH rather than "\
                          "using this flag."
      switch "--only-dependencies",
             description: "Install the dependencies with specified options but do not install the "\
                          "formula itself."
      flag   "--cc=",
             description: "Attempt to compile using the specified <compiler>, which should be the "\
                          "name of the compiler's executable, e.g. `gcc-7` for GCC 7. "\
                          "In order to use LLVM's clang, specify `llvm_clang`. To use the "\
                          "Apple-provided clang, specify `clang`. This option will only accept "\
                          "compilers that are provided by Homebrew or bundled with macOS. "\
                          "Please do not file issues if you encounter errors while using this option."
      switch "-s", "--build-from-source",
             description: "Compile <formula> from source even if a bottle is provided. "\
                          "Dependencies will still be installed from bottles if they are available."
      switch "--force-bottle",
             description: "Install from a bottle if it exists for the current or newest version of "\
                          "macOS, even if it would not normally be used for installation."
      switch "--include-test",
             description: "Install testing dependencies required to run `brew test` <formula>."
      switch "--devel",
             description: "If <formula> defines it, install the development version."
      switch "--HEAD",
             description: "If <formula> defines it, install the HEAD version, aka. master, trunk, unstable."
      switch "--fetch-HEAD",
             description: "Fetch the upstream repository to detect if the HEAD installation of the "\
                          "formula is outdated. Otherwise, the repository's HEAD will only be checked for "\
                          "updates when a new stable or development version has been released."
      switch "--keep-tmp",
             description: "Retain the temporary files created during installation."
      switch "--build-bottle",
             description: "Prepare the formula for eventual bottling during installation, skipping any "\
                          "post-install steps."
      flag   "--bottle-arch=",
             depends_on:  "--build-bottle",
             description: "Optimise bottles for the specified architecture rather than the oldest "\
                          "architecture supported by the version of macOS the bottles are built on."
      switch :force,
             description: "Install without checking for previously installed keg-only or "\
                          "non-migrated versions."
      switch :verbose,
             description: "Print the verification and postinstall steps."
      switch "--display-times",
             env:         :display_install_times,
             description: "Print install times for each formula at the end of the run."
      switch "-i", "--interactive",
             description: "Download and patch <formula>, then open a shell. This allows the user to "\
                          "run `./configure --help` and otherwise determine how to turn the software "\
                          "package into a Homebrew package."
      switch "-g", "--git",
             description: "Create a Git repository, useful for creating patches to the software."
      conflicts "--ignore-dependencies", "--only-dependencies"
      conflicts "--devel", "--HEAD"
      conflicts "--build-from-source", "--build-bottle", "--force-bottle"
      formula_options
    end
  end

  def install
    install_args.parse

    Homebrew.args.named.each do |name|
      next if File.exist?(name)
      next if name !~ HOMEBREW_TAP_FORMULA_REGEX && name !~ HOMEBREW_CASK_TAP_CASK_REGEX

      tap = Tap.fetch(Regexp.last_match(1), Regexp.last_match(2))
      tap.install unless tap.installed?
    end

    raise FormulaUnspecifiedError if args.remaining.empty?

    if args.ignore_dependencies?
      opoo <<~EOS
        #{Tty.bold}--ignore-dependencies is an unsupported Homebrew developer flag!#{Tty.reset}
        Adjust your PATH to put any preferred versions of applications earlier in the
        PATH rather than using this unsupported flag!

      EOS
    end

    formulae = []

    unless ARGV.casks.empty?
      cask_args = []
      cask_args << "--force" if args.force?
      cask_args << "--debug" if args.debug?
      cask_args << "--verbose" if args.verbose?

      ARGV.casks.each do |c|
        ohai "brew cask install #{c} #{cask_args.join " "}"
        system("#{HOMEBREW_PREFIX}/bin/brew", "cask", "install", c, *cask_args)
      end
    end

    # if the user's flags will prevent bottle only-installations when no
    # developer tools are available, we need to stop them early on
    FormulaInstaller.prevent_build_flags unless DevelopmentTools.installed?

    Homebrew.args.formulae.each do |f|
      # head-only without --HEAD is an error
      if !Homebrew.args.HEAD? && f.stable.nil? && f.devel.nil?
        raise <<~EOS
          #{f.full_name} is a head-only formula
          Install with `brew install --HEAD #{f.full_name}`
        EOS
      end

      # devel-only without --devel is an error
      if !args.devel? && f.stable.nil? && f.head.nil?
        raise <<~EOS
          #{f.full_name} is a devel-only formula
          Install with `brew install --devel #{f.full_name}`
        EOS
      end

      if !(args.HEAD? || args.devel?) && f.stable.nil?
        raise "#{f.full_name} has no stable download, please choose --devel or --HEAD"
      end

      # --HEAD, fail with no head defined
      raise "No head is defined for #{f.full_name}" if args.head? && f.head.nil?

      # --devel, fail with no devel defined
      raise "No devel block is defined for #{f.full_name}" if args.devel? && f.devel.nil?

      installed_head_version = f.latest_head_version
      if installed_head_version &&
         !f.head_version_outdated?(installed_head_version, fetch_head: args.fetch_HEAD?)
        new_head_installed = true
      end
      prefix_installed = f.prefix.exist? && !f.prefix.children.empty?

      if f.keg_only? && f.any_version_installed? && f.optlinked? && !args.force?
        # keg-only install is only possible when no other version is
        # linked to opt, because installing without any warnings can break
        # dependencies. Therefore before performing other checks we need to be
        # sure --force flag is passed.
        if f.outdated?
          optlinked_version = Keg.for(f.opt_prefix).version
          onoe <<~EOS
            #{f.full_name} #{optlinked_version} is already installed
            To upgrade to #{f.version}, run `brew upgrade #{f.name}`
          EOS
        elsif args.only_dependencies?
          formulae << f
        else
          opoo <<~EOS
            #{f.full_name} #{f.pkg_version} is already installed and up-to-date
            To reinstall #{f.pkg_version}, run `brew reinstall #{f.name}`
          EOS
        end
      elsif (args.HEAD? && new_head_installed) || prefix_installed
        # After we're sure that --force flag is passed for linked to opt
        # keg-only we need to be sure that the version we're attempting to
        # install is not already installed.

        installed_version = if args.HEAD?
          f.latest_head_version
        else
          f.pkg_version
        end

        msg = "#{f.full_name} #{installed_version} is already installed"
        linked_not_equals_installed = f.linked_version != installed_version
        if f.linked? && linked_not_equals_installed
          msg = <<~EOS
            #{msg}
            The currently linked version is #{f.linked_version}
            You can use `brew switch #{f} #{installed_version}` to link this version.
          EOS
        elsif !f.linked? || f.keg_only?
          msg = <<~EOS
            #{msg}, it's just not linked
            You can use `brew link #{f}` to link this version.
          EOS
        elsif args.only_dependencies?
          msg = nil
          formulae << f
        else
          msg = <<~EOS
            #{msg} and up-to-date
            To reinstall #{f.pkg_version}, run `brew reinstall #{f.name}`
          EOS
        end
        opoo msg if msg
      elsif !f.any_version_installed? && old_formula = f.old_installed_formulae.first
        msg = "#{old_formula.full_name} #{old_formula.installed_version} already installed"
        if !old_formula.linked? && !old_formula.keg_only?
          msg = <<~EOS
            #{msg}, it's just not linked.
            You can use `brew link #{old_formula.full_name}` to link this version.
          EOS
        end
        opoo msg
      elsif f.migration_needed? && !args.force?
        # Check if the formula we try to install is the same as installed
        # but not migrated one. If --force is passed then install anyway.
        opoo <<~EOS
          #{f.oldname} is already installed, it's just not migrated
          You can migrate this formula with `brew migrate #{f}`
          Or you can force install it with `brew install #{f} --force`
        EOS
      else
        # If none of the above is true and the formula is linked, then
        # FormulaInstaller will handle this case.
        formulae << f
      end

      # Even if we don't install this formula mark it as no longer just
      # installed as a dependency.
      next unless f.opt_prefix.directory?

      keg = Keg.new(f.opt_prefix.resolved_path)
      tab = Tab.for_keg(keg)
      unless tab.installed_on_request
        tab.installed_on_request = true
        tab.write
      end
    end

    return if formulae.empty?

    Install.perform_preinstall_checks

    formulae.each do |f|
      Migrator.migrate_if_needed(f)
      install_formula(f)
      Cleanup.install_formula_clean!(f)
    end
    Homebrew.messages.display_messages
  rescue FormulaUnreadableError, FormulaClassUnavailableError,
         TapFormulaUnreadableError, TapFormulaClassUnavailableError => e
    # Need to rescue before `FormulaUnavailableError` (superclass of this)
    # is handled, as searching for a formula doesn't make sense here (the
    # formula was found, but there's a problem with its implementation).
    ofail e.message
  rescue FormulaUnavailableError => e
    if e.name == "updog"
      ofail "What's updog?"
      return
    end

    ofail e.message
    if (reason = MissingFormula.reason(e.name))
      $stderr.puts reason
      return
    end

    ohai "Searching for similarly named formulae..."
    formulae_search_results = search_formulae(e.name)
    case formulae_search_results.length
    when 0
      ofail "No similarly named formulae found."
    when 1
      puts "This similarly named formula was found:"
      puts formulae_search_results
      puts "To install it, run:\n  brew install #{formulae_search_results.first}"
    else
      puts "These similarly named formulae were found:"
      puts Formatter.columns(formulae_search_results)
      puts "To install one of them, run (for example):\n  brew install #{formulae_search_results.first}"
    end

    # Do not search taps if the formula name is qualified
    return if e.name.include?("/")

    ohai "Searching taps..."
    taps_search_results = search_taps(e.name)[:formulae]
    case taps_search_results.length
    when 0
      ofail "No formulae found in taps."
    when 1
      puts "This formula was found in a tap:"
      puts taps_search_results
      puts "To install it, run:\n  brew install #{taps_search_results.first}"
    else
      puts "These formulae were found in taps:"
      puts Formatter.columns(taps_search_results)
      puts "To install one of them, run (for example):\n  brew install #{taps_search_results.first}"
    end
  end

  def install_formula(f)
    f.print_tap_action
    build_options = f.build

    fi = FormulaInstaller.new(f)
    fi.options              = build_options.used_options
    fi.ignore_deps          = args.ignore_dependencies?
    fi.only_deps            = args.only_dependencies?
    fi.build_bottle         = args.build_bottle?
    fi.interactive          = args.interactive?
    fi.git                  = args.git?
    fi.prelude
    fi.install
    fi.finish
  rescue FormulaInstallationAlreadyAttemptedError
    # We already attempted to install f as part of the dependency tree of
    # another formula. In that case, don't generate an error, just move on.
    nil
  rescue CannotInstallFormulaError => e
    ofail e.message
  end
end
