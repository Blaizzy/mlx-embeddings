# frozen_string_literal: true

require "formula"
require "keg"
require "cli/parser"
require "cask/cmd"
require "cask/caskroom"

module Homebrew
  module_function

  def outdated_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `outdated` [<options>] [<formula>]

        List installed formulae that have an updated version available. By default, version
        information is displayed in interactive shells, and suppressed otherwise.
      EOS
      switch :quiet,
             description: "List only the names of outdated kegs (takes precedence over `--verbose`)."
      switch :verbose,
             description: "Include detailed version information."
      flag   "--json",
             description: "Print output in JSON format. Currently the default and only accepted "\
                          "value for <version> is `v1`. See the docs for examples of using the JSON "\
                          "output: <https://docs.brew.sh/Querying-Brew>. "\
                          "By default, this option treats all arguments as formulae. "\
                          "To treat arguments as casks, use the --cask-only option."
      switch "--fetch-HEAD",
             description: "Fetch the upstream repository to detect if the HEAD installation of the "\
                          "formula is outdated. Otherwise, the repository's HEAD will only be checked for "\
                          "updates when a new stable or development version has been released."
      switch "--greedy",
             description: "Print outdated casks with `auto_updates` or `version :latest`"
      switch "--formula-only",
             description: "Treat all arguments as formulae"
      switch "--cask-only",
             description: "Treat all arguments as casks"
      switch :debug
      conflicts "--quiet", "--verbose", "--json"
      conflicts "--formula-only", "--cask-only"
    end
  end

  def outdated
    outdated_args.parse

    if args.json
      raise UsageError, "invalid JSON version: #{args.json}" unless ["v1", true].include? args.json
    end

    if args.formula_only? || !args.cask_only? && args.json
      formulae = args.resolved_formulae.blank? ? Formula.installed : args.resolved_formulae
      outdated = print_outdated_formulae(formulae)
    elsif args.cask_only?
      casks = args.named.blank? ? Cask::Caskroom.casks : args.named
      outdated = print_outdated_casks(casks)
    else
      formulae, casks = args.resolved_formulae_casks

      if formulae.blank? && casks.blank?
        formulae = Formula.installed
        casks = Cask::Caskroom.casks
      end

      outdated = print_outdated_formulae(formulae) + print_outdated_casks(casks)
    end

    Homebrew.failed = args.named.present? && outdated.present?
  end

  def print_outdated_formulae(formulae)
    return print_outdated_formulae_json(formulae) if args.json

    fetch_head = args.fetch_HEAD?

    outdated_formulae = formulae.select { |f| f.outdated?(fetch_head: fetch_head) }
                                .sort

    outdated_formulae.each do |f|
      if verbose?
        outdated_kegs = f.outdated_kegs(fetch_head: fetch_head)

        current_version = if f.alias_changed?
          latest = f.latest_formula
          "#{latest.name} (#{latest.pkg_version})"
        elsif f.head? && outdated_kegs.any? { |k| k.version.to_s == f.pkg_version.to_s }
          # There is a newer HEAD but the version number has not changed.
          "latest HEAD"
        else
          f.pkg_version.to_s
        end

        outdated_versions = outdated_kegs
                            .group_by { |keg| Formulary.from_keg(keg).full_name }
                            .sort_by { |full_name, _kegs| full_name }
                            .map do |full_name, kegs|
          "#{full_name} (#{kegs.map(&:version).join(", ")})"
        end.join(", ")

        pinned_version = " [pinned at #{f.pinned_version}]" if f.pinned?

        puts "#{outdated_versions} < #{current_version}#{pinned_version}"
      else
        puts f.full_installed_specified_name
      end
    end
  end

  def print_outdated_formulae_json(formulae)
    json = []
    fetch_head = args.fetch_HEAD?
    outdated_formulae = formulae.select { |f| f.outdated?(fetch_head: fetch_head) }

    outdated = outdated_formulae.each do |f|
      outdated_versions = f.outdated_kegs(fetch_head: fetch_head).map(&:version)
      current_version = if f.head? && outdated_versions.any? { |v| v.to_s == f.pkg_version.to_s }
        "HEAD"
      else
        f.pkg_version.to_s
      end

      json << { name:               f.full_name,
                installed_versions: outdated_versions.map(&:to_s),
                current_version:    current_version,
                pinned:             f.pinned?,
                pinned_version:     f.pinned_version }
    end
    puts JSON.generate(json)

    outdated
  end

  def print_outdated_casks(casks)
    outdated = casks.map { |cask| Cask::CaskLoader.load(cask) }.select do |cask|
      odebug "Checking update info of Cask #{cask}"
      cask.outdated?(args.greedy?)
    end

    output = outdated.map { |cask| cask.outdated_info(args.greedy?, verbose?, args.json) }
    puts args.json ? JSON.generate(output) : output

    outdated
  end

  def verbose?
    ($stdout.tty? || args.verbose?) && !args.quiet?
  end
end
