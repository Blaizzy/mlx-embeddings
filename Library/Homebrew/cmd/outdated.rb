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
             description: "Print output in JSON format. There are two versions: v1 and v2. " \
                          "v1 is deprecated and is currently the default if no version is specified. " \
                          "v2 prints outdated formulae and casks. "
      switch "--fetch-HEAD",
             description: "Fetch the upstream repository to detect if the HEAD installation of the "\
                          "formula is outdated. Otherwise, the repository's HEAD will only be checked for "\
                          "updates when a new stable or development version has been released."
      switch "--greedy",
             description: "Print outdated casks with `auto_updates` or `version :latest`"
      switch "--formula",
             description: "Treat all arguments as formulae"
      switch "--cask",
             description: "Treat all arguments as casks"
      switch :debug
      conflicts "--quiet", "--verbose", "--json"
      conflicts "--formula", "--cask"
    end
  end

  def outdated
    outdated_args.parse

    case json_version
    when :v1
      opoo "JSON v1 has been deprecated. Please use --json=v2"

      outdated = if args.formula? || !args.cask?
        outdated_formulae
      else
        outdated_casks
      end

      puts JSON.generate(json_info(outdated))

    when :v2
      formulae, casks = if args.formula?
        [outdated_formulae, []]
      elsif args.cask?
        [[], outdated_casks]
      else
        outdated_formulae_casks
      end

      puts JSON.generate({
                           "formulae" => json_info(formulae),
                           "casks"    => json_info(casks),
                         })

      outdated = formulae + casks

    else
      outdated = if args.formula?
        outdated_formulae
      elsif args.cask?
        outdated_casks
      else
        outdated_formulae_casks.flatten
      end

      print_outdated(outdated)
    end

    Homebrew.failed = args.named.present? && outdated.present?
  end

  def print_outdated(formula_or_cask)
    return formula_or_cask.each { |f_or_c| print_outdated(f_or_c) } if formula_or_cask.is_a? Array

    if formula_or_cask.is_a?(Formula)
      f = formula_or_cask

      if verbose?
        outdated_kegs = f.outdated_kegs(fetch_head: args.fetch_HEAD?)

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
    else
      c = formula_or_cask

      puts c.outdated_info(args.greedy?, verbose?, false)
    end
  end

  def json_info(formula_or_cask)
    return formula_or_cask.map { |f_or_c| json_info(f_or_c) } if formula_or_cask.is_a? Array

    if formula_or_cask.is_a?(Formula)
      f = formula_or_cask

      outdated_versions = f.outdated_kegs(fetch_head: args.fetch_HEAD?).map(&:version)
      current_version = if f.head? && outdated_versions.any? { |v| v.to_s == f.pkg_version.to_s }
        "HEAD"
      else
        f.pkg_version.to_s
      end

      { name:               f.full_name,
        installed_versions: outdated_versions.map(&:to_s),
        current_version:    current_version,
        pinned:             f.pinned?,
        pinned_version:     f.pinned_version }
    else
      c = formula_or_cask

      c.outdated_info(args.greedy?, verbose?, true)
    end
  end

  def verbose?
    ($stdout.tty? || args.verbose?) && !args.quiet?
  end

  def json_version
    version_hash = {
      nil  => nil,
      true => :v1,
      "v1" => :v1,
      "v2" => :v2,
    }

    raise UsageError, "invalid JSON version: #{args.json}" unless version_hash.include? args.json

    version_hash[args.json]
  end

  def outdated_formulae
    select_outdated((args.resolved_formulae.presence || Formula.installed)).sort
  end

  def outdated_casks
    select_outdated(
      args.named.present? ? args.named.uniq.map { |ref| Cask::CaskLoader.load ref } : Cask::Caskroom.casks,
    )
  end

  def outdated_formulae_casks
    formulae, casks = args.resolved_formulae_casks

    if formulae.blank? && casks.blank?
      formulae = Formula.installed
      casks = Cask::Caskroom.casks
    end

    [select_outdated(formulae), select_outdated(casks)]
  end

  def select_outdated(formulae_or_casks)
    formulae_or_casks.select do |fc|
      fc.is_a?(Formula) ? fc.outdated?(fetch_head: args.fetch_HEAD?) : fc.outdated?(args.greedy?)
    end
  end
end
