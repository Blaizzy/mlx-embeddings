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
      switch "-r", "--resources",
             description: "Check resources with livecheck blocks."
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

    package_and_resource_to_check = if args.tap
      tap = Tap.fetch(args.tap)
      formulae = args.cask? ? [] : tap.formula_files.map { |path| Formulary.factory(path) }
      casks = args.formula? ? [] : tap.cask_files.map { |path| Cask::CaskLoader.load(path) }
      formulae + casks
    elsif args.installed?
      formulae = args.cask? ? [] : Formula.installed
      casks = args.formula? ? [] : Cask::Caskroom.casks
      formulae + casks
    elsif args.resources?
      formula_with_resources = Formula.all.select { |formula| formula.resources.any? }
      formula_with_resources
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
        args.named.to_package_and_resource
      end
    elsif File.exist?(WATCHLIST_PATH)
      begin
        names = Pathname.new(WATCHLIST_PATH).read.lines
                        .reject { |line| line.start_with?("#") || line.blank? }
                        .map(&:strip)

        named_args = T.unsafe(CLI::NamedArgs).new(*names, parent: args)
        named_args.to_package_and_resource(ignore_unavailable: true)
      rescue Errno::ENOENT => e
        onoe e
      end
    else
      raise UsageError, "A watchlist file is required when no arguments are given."
    end

    # p package_and_resource_to_check.class
    # p package_and_resource_to_check.length
    # p package_and_resource_to_check[0].class
    # p package_and_resource_to_check.map { |d| d.name }

    package_and_resource_to_check = package_and_resource_to_check.sort_by do |package_or_resource|
      package_or_resource.respond_to?(:token) ? package_or_resource.token : package_or_resource.name
    end

    raise UsageError, "No formulae or casks to check." if package_and_resource_to_check.blank?

    options = {
      json:                 args.json?,
      full_name:            args.full_name?,
      handle_name_conflict: !args.formula? && !args.cask?,
      newer_only:           args.newer_only?,
      quiet:                args.quiet?,
      debug:                args.debug?,
      verbose:              args.verbose?,
    }.compact

    Livecheck.run_checks(package_and_resource_to_check[1..3], **options)
  end
end
