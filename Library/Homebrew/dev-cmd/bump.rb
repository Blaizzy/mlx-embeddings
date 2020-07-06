# frozen_string_literal: true

require "cli/parser"

module Homebrew
  module_function

  def bump_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `bump`

        Display out-of-date brew formulae, the latest version available, and whether a pull request has been opened.
      EOS
      switch :verbose
      switch :debug
    end
  end

  def bump
    bump_args.parse

    outdated_repology_packages = Repology.parse_api_response
    outdated_packages = validate_and_format_packages(outdated_repology_packages)

    display(outdated_packages)
  end

  def validate_and_format_packages(outdated_repology_packages)
    ohai "Verifying outdated repology packages as Homebrew Formulae"

    packages = {}
    outdated_repology_packages.each do |_name, repositories|
      # identify homebrew repo
      repology_homebrew_repo = repositories.find do |repo|
        repo["repo"] == "homebrew"
      end

      next if repology_homebrew_repo.empty?

      latest_version = nil

      # identify latest version amongst repology repos
      repositories.each do |repo|
        latest_version = repo["version"] if repo["status"] == "newest"
      end

      packages[repology_homebrew_repo["srcname"]] = format_package(repology_homebrew_repo["srcname"], latest_version)
    end
    packages
  end

  def format_package(package_name, latest_version)
    formula = get_formula_details(package_name)
    
    return if formula.nil?

    tap_full_name = formula.tap&.full_name
    current_version = current_formula_version(formula)
    livecheck_response = livecheck_formula(package_name)
    pull_requests = GitHub.check_for_duplicate_pull_requests(formula, tap_full_name, latest_version, args, true)

    {
      repology_latest_version:  latest_version,
      current_formula_version:  current_version.to_s,
      livecheck_latest_version: livecheck_response[:livecheck_version],
      open_pull_requests:       pull_requests,
    }
  end

  def get_formula_details(formula_name)
    Formula[formula_name]
  rescue
    nil
  end

  def current_formula_version(formula)
    formula.version.to_s
  rescue
    nil
  end

  def livecheck_formula(formula)
    ohai "Checking livecheck formula: #{formula}" if Homebrew.args.verbose?

    response = Utils.popen_read(HOMEBREW_BREW_FILE, "livecheck", formula, "--quiet").chomp

    parse_livecheck_response(response)
  end

  def parse_livecheck_response(response)
    output = response.delete(" ").split(/:|==>/)

    # eg: ["openclonk", "7.0", "8.1"]
    package_name, brew_version, latest_version = output

    {
      name:              package_name,
      formula_version:   brew_version,
      livecheck_version: latest_version,
    }
  end

  def display(outdated_packages)
    ohai "Outdated Formulae\n"

    outdated_packages.each do |formula, package_details|
      ohai formula
      ohai "Current formula version: #{package_details[:current_formula_version]}"
      ohai "Latest repology version: #{package_details[:repology_latest_version]}"
      ohai "Latest livecheck version: #{package_details[:livecheck_latest_version]}"
      ohai "Open pull requests: #{package_details[:open_pull_requests]}"
    end
  end
end
