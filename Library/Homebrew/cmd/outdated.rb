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
      switch "-q", "--quiet",
             description: "List only the names of outdated kegs (takes precedence over `--verbose`)."
      switch "-v", "--verbose",
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
             description: "Print outdated casks with `auto_updates` or `version :latest`."
      switch "--formula",
             description: "Treat all arguments as formulae."
      switch "--cask",
             description: "Treat all arguments as casks."

      conflicts "--quiet", "--verbose", "--json"
      conflicts "--formula", "--cask"
    end
  end

  def outdated
    args = outdated_args.parse

    case json_version(args.json)
    when :v1, :default
      # TODO: enable for next major/minor release
      # odeprecated "brew outdated --json#{json_version == :v1 ? "=v1" : ""}", "brew outdated --json=v2"

      outdated = if args.formula? || !args.cask?
        outdated_formulae
      else
        outdated_casks
      end

      puts JSON.generate(json_info(outdated, args: args))

    when :v2
      formulae, casks = if args.formula?
        [outdated_formulae, []]
      elsif args.cask?
        [[], outdated_casks]
      else
        outdated_formulae_casks
      end

      json = {
        "formulae" => json_info(formulae, args: args),
        "casks"    => json_info(casks, args: args),
      }
      puts JSON.generate(json)

      outdated = formulae + casks

    else
      outdated = if args.formula?
        outdated_formulae
      elsif args.cask?
        outdated_casks
      else
        outdated_formulae_casks.flatten
      end

      print_outdated(outdated, args: args)
    end

    Homebrew.failed = args.named.present? && outdated.present?
  end

  def print_outdated(formulae_or_casks, args:)
    formulae_or_casks.each do |formula_or_cask|
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

          outdated_versions = outdated_kegs.group_by { |keg| Formulary.from_keg(keg).full_name }
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
  end

  def json_info(formulae_or_casks, args:)
    formulae_or_casks.map do |formula_or_cask|
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
  end

  def verbose?
    ($stdout.tty? || args.verbose?) && !args.quiet?
  end

  def json_version(version)
    version_hash = {
      nil  => nil,
      true => :default,
      "v1" => :v1,
      "v2" => :v2,
    }

    raise UsageError, "invalid JSON version: #{version}" unless version_hash.include?(version)

    version_hash[version]
  end

  def outdated_formulae
    select_outdated((args.resolved_formulae.presence || Formula.installed)).sort
  end

  def outdated_casks
    if args.named.present?
      select_outdated(args.named.uniq.map(&Cask::CaskLoader.method(:load)))
    else
      select_outdated(Cask::Caskroom.casks)
    end
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
    formulae_or_casks.select do |formula_or_cask|
      if formula_or_cask.is_a?(Formula)
        formula_or_cask.outdated?(fetch_head: args.fetch_HEAD?)
      else
        formula_or_cask.outdated?(args.greedy?)
      end
    end
  end
end
