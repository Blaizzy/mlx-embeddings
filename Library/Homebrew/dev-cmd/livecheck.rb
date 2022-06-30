# typed: true
# frozen_string_literal: true

require "cli/parser"
require "formula"
require "livecheck/livecheck"
require "livecheck/strategy"

module Homebrew
  extend T::Sig

  module_function

  WATCHLIST_PATH = File.expand_path(Homebrew::EnvConfig.livecheck_watchlist).freeze

  sig { returns(CLI::Parser) }
  def livecheck_args
    Homebrew::CLI::Parser.new do
      description <<~EOS
        Check for newer versions of formulae and/or casks from upstream.

        If no formula or cask argument is passed, the list of formulae and
        casks to check is taken from `HOMEBREW_LIVECHECK_WATCHLIST` or
        `~/.brew_livecheck_watchlist`.
      EOS
      switch "--full-name",
             description: "Print formulae/casks with fully-qualified names."
      flag   "--tap=",
             description: "Check formulae/casks within the given tap, specified as <user>`/`<repo>."
      switch "--all",
             description: "Check all available formulae/casks."
      switch "--installed",
             description: "Check formulae/casks that are currently installed."
      switch "--newer-only",
             description: "Show the latest version only if it's newer than the formula/cask."
      switch "--json",
             description: "Output information in JSON format."
      switch "-q", "--quiet",
             description: "Suppress warnings, don't print a progress bar for JSON output."
      switch "--formula", "--formulae",
             description: "Only check formulae."
      switch "--cask", "--casks",
             description: "Only check casks."

      conflicts "--debug", "--json"
      conflicts "--tap=", "--all", "--installed"
      conflicts "--cask", "--formula"

      named_args [:formula, :cask]
    end
  end

  def livecheck
    args = livecheck_args.parse

    if args.debug? && args.verbose?
      puts args
      puts Homebrew::EnvConfig.livecheck_watchlist if Homebrew::EnvConfig.livecheck_watchlist.present?
    end

    formulae_and_casks_to_check = if args.tap
      tap = Tap.fetch(args.tap)
      formulae = args.cask? ? [] : tap.formula_files.map { |path| Formulary.factory(path) }
      casks = args.formula? ? [] : tap.cask_files.map { |path| Cask::CaskLoader.load(path) }
      formulae + casks
    elsif args.installed?
      formulae = args.cask? ? [] : Formula.installed
      casks = args.formula? ? [] : Cask::Caskroom.casks
      formulae + casks
    elsif args.all?
      formulae = args.cask? ? [] : Formula.all
      casks = args.formula? ? [] : Cask::Cask.all
      formulae + casks
    elsif args.named.present?
      if args.formula?
        args.named.to_formulae
      elsif args.cask?
        args.named.to_casks
      else
        args.named.to_formulae_and_casks
      end
    elsif File.exist?(WATCHLIST_PATH)
      begin
        names = Pathname.new(WATCHLIST_PATH).read.lines
                        .reject { |line| line.start_with?("#") || line.blank? }
                        .map(&:strip)

        named_args = T.unsafe(CLI::NamedArgs).new(*names, parent: args)
        named_args.to_formulae_and_casks(ignore_unavailable: true)
      rescue Errno::ENOENT => e
        onoe e
      end
    else
      raise UsageError, "A watchlist file is required when no arguments are given."
    end
    formulae_and_casks_to_check = formulae_and_casks_to_check.sort_by do |formula_or_cask|
      formula_or_cask.respond_to?(:token) ? formula_or_cask.token : formula_or_cask.name
    end

    raise UsageError, "No formulae or casks to check." if formulae_and_casks_to_check.blank?

    options = {
      json:                 args.json?,
      full_name:            args.full_name?,
      handle_name_conflict: !args.formula? && !args.cask?,
      newer_only:           args.newer_only?,
      quiet:                args.quiet?,
      debug:                args.debug?,
      verbose:              args.verbose?,
    }.compact

    Livecheck.run_checks(formulae_and_casks_to_check, **options)
  end
end
