# typed: true
# frozen_string_literal: true

require "cli/parser"
require "formula"
require "livecheck/livecheck"
require "livecheck/strategy"

module Homebrew
  extend T::Sig

  module_function

  WATCHLIST_PATH = (
    ENV["HOMEBREW_LIVECHECK_WATCHLIST"] ||
    "#{Dir.home}/.brew_livecheck_watchlist"
  ).freeze

  sig { returns(CLI::Parser) }
  def livecheck_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `livecheck` [<formulae>|<casks>]

        Check for newer versions of formulae and/or casks from upstream.

        If no formula or cask argument is passed, the list of formulae and casks to check is taken from
        `HOMEBREW_LIVECHECK_WATCHLIST` or `~/.brew_livecheck_watchlist`.
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
    end
  end

  def livecheck
    args = livecheck_args.parse

    if args.debug? && args.verbose?
      puts args
      puts ENV["HOMEBREW_LIVECHECK_WATCHLIST"] if ENV["HOMEBREW_LIVECHECK_WATCHLIST"].present?
    end

    formulae_and_casks_to_check = if args.tap
      tap = Tap.fetch(args.tap)
      formulae = !args.cask? ? tap.formula_names.map { |name| Formula[name] } : []
      casks = !args.formula? ? tap.cask_tokens.map { |token| Cask::CaskLoader.load(token) } : []
      formulae + casks
    elsif args.installed?
      formulae = !args.cask? ? Formula.installed : []
      casks = !args.formula? ? Cask::Caskroom.casks : []
      formulae + casks
    elsif args.all?
      formulae = !args.cask? ? Formula.to_a : []
      casks = !args.formula? ? Cask::Cask.to_a : []
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
        named_args = CLI::NamedArgs.new(*names)
        if args.formula?
          named_args.to_formulae
        elsif args.cask?
          named_args.to_casks
        else
          named_args.to_formulae_and_casks
        end
      rescue Errno::ENOENT => e
        onoe e
      end
    end.sort_by do |formula_or_cask|
      formula_or_cask.respond_to?("token") ? formula_or_cask.token : formula_or_cask.name
    end

    raise UsageError, "No formulae or casks to check." if formulae_and_casks_to_check.blank?

    Livecheck.run_checks(formulae_and_casks_to_check, args)
  end
end
