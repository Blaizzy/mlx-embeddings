# typed: false
# frozen_string_literal: true

require "cask/config"
require "cask/cmd"
require "cask/cmd/install"
require "missing_formula"
require "formula_installer"
require "development_tools"
require "install"
require "search"
require "cleanup"
require "cli/parser"
require "upgrade"

module Homebrew
  extend T::Sig

  extend Search

  module_function

  sig { returns(CLI::Parser) }
  def install_args
    Homebrew::CLI::Parser.new do
      description <<~EOS
        Install a <formula> or <cask>. Additional options specific to a <formula> may be
        appended to the command.

        Unless `HOMEBREW_NO_INSTALL_CLEANUP` is set, `brew cleanup` will then be run for
        the installed formulae or, every 30 days, for all formulae.
      EOS
      switch "-d", "--debug",
             description: "If brewing fails, open an interactive debugging session with access to IRB " \
                          "or a shell inside the temporary build directory."
      switch "-f", "--force",
             description: "Install formulae without checking for previously installed keg-only or " \
                          "non-migrated versions. When installing casks, overwrite existing files "\
                          "(binaries and symlinks are excluded, unless originally from the same cask)."
      switch "-v", "--verbose",
             description: "Print the verification and postinstall steps."
      [
        [:switch, "--formula", "--formulae", {
          description: "Treat all named arguments as formulae.",
        }],
        [:flag, "--env=", {
          description: "Disabled other than for internal Homebrew use.",
        }],
        [:switch, "--ignore-dependencies", {
          description: "An unsupported Homebrew development flag to skip installing any dependencies of any kind. " \
                       "If the dependencies are not already present, the formula will have issues. If you're not " \
                       "developing Homebrew, consider adjusting your PATH rather than using this flag.",
        }],
        [:switch, "--only-dependencies", {
          description: "Install the dependencies with specified options but do not install the " \
                       "formula itself.",
        }],
        [:flag, "--cc=", {
          description: "Attempt to compile using the specified <compiler>, which should be the name of the " \
                       "compiler's executable, e.g. `gcc-7` for GCC 7. In order to use LLVM's clang, specify " \
                       "`llvm_clang`. To use the Apple-provided clang, specify `clang`. This option will only " \
                       "accept compilers that are provided by Homebrew or bundled with macOS. Please do not " \
                       "file issues if you encounter errors while using this option.",
        }],
        [:switch, "-s", "--build-from-source", {
          description: "Compile <formula> from source even if a bottle is provided. " \
                       "Dependencies will still be installed from bottles if they are available.",
        }],
        [:switch, "--force-bottle", {
          description: "Install from a bottle if it exists for the current or newest version of " \
                       "macOS, even if it would not normally be used for installation.",
        }],
        [:switch, "--include-test", {
          description: "Install testing dependencies required to run `brew test` <formula>.",
        }],
        [:switch, "--HEAD", {
          description: "If <formula> defines it, install the HEAD version, aka. main, trunk, unstable, master.",
        }],
        [:switch, "--fetch-HEAD", {
          description: "Fetch the upstream repository to detect if the HEAD installation of the " \
                       "formula is outdated. Otherwise, the repository's HEAD will only be checked for " \
                       "updates when a new stable or development version has been released.",
        }],
        [:switch, "--keep-tmp", {
          description: "Retain the temporary files created during installation.",
        }],
        [:switch, "--build-bottle", {
          description: "Prepare the formula for eventual bottling during installation, skipping any " \
                       "post-install steps.",
        }],
        [:flag, "--bottle-arch=", {
          depends_on:  "--build-bottle",
          description: "Optimise bottles for the specified architecture rather than the oldest " \
                       "architecture supported by the version of macOS the bottles are built on.",
        }],
        [:switch, "--display-times", {
          env:         :display_install_times,
          description: "Print install times for each package at the end of the run.",
        }],
        [:switch, "-i", "--interactive", {
          description: "Download and patch <formula>, then open a shell. This allows the user to " \
                       "run `./configure --help` and otherwise determine how to turn the software " \
                       "package into a Homebrew package.",
        }],
        [:switch, "-g", "--git", {
          description: "Create a Git repository, useful for creating patches to the software.",
        }],
      ].each do |*args, **options|
        send(*args, **options)
        conflicts "--cask", args.last
      end
      formula_options
      [
        [:switch, "--cask", "--casks", { description: "Treat all named arguments as casks." }],
        *Cask::Cmd::AbstractCommand::OPTIONS,
        *Cask::Cmd::Install::OPTIONS,
      ].each do |*args, **options|
        send(*args, **options)
        conflicts "--formula", args.last
      end
      cask_options

      conflicts "--ignore-dependencies", "--only-dependencies"
      conflicts "--build-from-source", "--build-bottle", "--force-bottle"

      named_args [:formula, :cask], min: 1
    end
  end

  def install
    args = install_args.parse

    if args.env.present?
      # Can't use `replacement: false` because `install_args` are used by
      # `build.rb`. Instead, `hide_from_man_page` and don't do anything with
      # this argument here.
      odisabled "brew install --env", "`env :std` in specific formula files"
    end

    args.named.each do |name|
      next if File.exist?(name)
      next if name !~ HOMEBREW_TAP_FORMULA_REGEX && name !~ HOMEBREW_CASK_TAP_CASK_REGEX

      tap = Tap.fetch(Regexp.last_match(1), Regexp.last_match(2))
      tap.install unless tap.installed?
    end

    if args.ignore_dependencies?
      opoo <<~EOS
        #{Tty.bold}`--ignore-dependencies` is an unsupported Homebrew developer flag!#{Tty.reset}
        Adjust your PATH to put any preferred versions of applications earlier in the
        PATH rather than using this unsupported flag!

      EOS
    end

    begin
      formulae, casks = args.named.to_formulae_and_casks(prefer_loading_from_api: true)
                            .partition { |formula_or_cask| formula_or_cask.is_a?(Formula) }
    rescue FormulaOrCaskUnavailableError, Cask::CaskUnavailableError => e
      retry if Tap.install_default_cask_tap_if_necessary(force: args.cask?)

      raise e
    end

    if casks.any?
      Cask::Cmd::Install.install_casks(
        *casks,
        binaries:       args.binaries?,
        verbose:        args.verbose?,
        force:          args.force?,
        require_sha:    args.require_sha?,
        skip_cask_deps: args.skip_cask_deps?,
        quarantine:     args.quarantine?,
      )
    end

    # if the user's flags will prevent bottle only-installations when no
    # developer tools are available, we need to stop them early on
    unless DevelopmentTools.installed?
      build_flags = []

      build_flags << "--HEAD" if args.HEAD?
      build_flags << "--build-bottle" if args.build_bottle?
      build_flags << "--build-from-source" if args.build_from_source?

      raise BuildFlagsError.new(build_flags, bottled: formulae.all?(&:bottled?)) if build_flags.present?
    end

    installed_formulae = formulae.select do |f|
      Install.install_formula?(
        f,
        head:              args.HEAD?,
        fetch_head:        args.fetch_HEAD?,
        only_dependencies: args.only_dependencies?,
        force:             args.force?,
        quiet:             args.quiet?,
      )
    end

    return if installed_formulae.empty?

    Install.perform_preinstall_checks(cc: args.cc)

    Install.install_formulae(
      installed_formulae,
      build_bottle:               args.build_bottle?,
      force_bottle:               args.force_bottle?,
      bottle_arch:                args.bottle_arch,
      ignore_deps:                args.ignore_dependencies?,
      only_deps:                  args.only_dependencies?,
      include_test_formulae:      args.include_test_formulae,
      build_from_source_formulae: args.build_from_source_formulae,
      cc:                         args.cc,
      git:                        args.git?,
      interactive:                args.interactive?,
      keep_tmp:                   args.keep_tmp?,
      force:                      args.force?,
      debug:                      args.debug?,
      quiet:                      args.quiet?,
      verbose:                    args.verbose?,
    )

    Upgrade.check_installed_dependents(
      installed_formulae,
      flags:                      args.flags_only,
      installed_on_request:       args.named.present?,
      force_bottle:               args.force_bottle?,
      build_from_source_formulae: args.build_from_source_formulae,
      interactive:                args.interactive?,
      keep_tmp:                   args.keep_tmp?,
      force:                      args.force?,
      debug:                      args.debug?,
      quiet:                      args.quiet?,
      verbose:                    args.verbose?,
    )

    Homebrew.messages.display_messages(display_times: args.display_times?)
  rescue FormulaUnreadableError, FormulaClassUnavailableError,
         TapFormulaUnreadableError, TapFormulaClassUnavailableError => e
    # Need to rescue before `FormulaUnavailableError` (superclass of this)
    # is handled, as searching for a formula doesn't make sense here (the
    # formula was found, but there's a problem with its implementation).
    $stderr.puts e.backtrace if Homebrew::EnvConfig.developer?
    ofail e.message
  rescue FormulaOrCaskUnavailableError => e
    if e.name == "updog"
      ofail "What's updog?"
      return
    end

    opoo e
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

    if (reason = MissingFormula.reason(e.name))
      $stderr.puts reason
      return
    end

    # Do not search taps if the formula name is qualified
    return if e.name.include?("/")

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
end
