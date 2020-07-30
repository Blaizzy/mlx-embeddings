# frozen_string_literal: true

require "cli/parser"

module Homebrew
  module_function

  def bump_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `bump` [<options>]

        Display out-of-date brew formulae and the latest version available.
        Also displays whether a pull request has been opened with the URL.
      EOS
      flag "--formula=",
           description: "Return results for package by name."
      flag "--limit=",
           description: "Limit number of package results returned."
      switch :verbose
      switch :debug
    end
  end

  def bump
    bump_args.parse

    requested_formula = Homebrew.args.formula
    requested_formula&.downcase!

    raise FormulaUnavailableError, requested_formula if requested_formula && !get_formula_details(requested_formula)

    outdated_repology_packages = if requested_formula
      Repology.single_package_query(requested_formula)
    else
      Repology.parse_api_response
    end

    if requested_formula && outdated_repology_packages.nil?
      ohai "#{requested_formula} is up-to-date!"
      puts "Current version: #{get_formula_details(requested_formula).version}"
      return
    end

    outdated_packages = validate_and_format_packages(outdated_repology_packages)
    display(outdated_packages)
  end

  def validate_and_format_packages(outdated_repology_packages)
    packages = {}
    outdated_repology_packages.each do |_name, repositories|
      # identify homebrew repo
      repology_homebrew_repo = repositories.find do |repo|
        repo["repo"] == "homebrew"
      end

      next if repology_homebrew_repo.blank?

      latest_version = repositories.find { |repo| repo["status"] == "newest" }["version"]
      srcname = repology_homebrew_repo["srcname"]
      package_details = format_package(srcname, latest_version)
      packages[srcname] = package_details unless package_details.nil?

      break if Homebrew.args.limit && packages.size >= Homebrew.args.limit.to_i
    end

    packages
  end

  def format_package(package_name, latest_version)
    formula = get_formula_details(package_name)

    return if formula.blank?

    tap_full_name = formula.tap&.full_name
    current_version = current_formula_version(formula)
    livecheck_response = livecheck_formula(package_name)
    pull_requests = GitHub.check_for_duplicate_pull_requests(formula, tap_full_name, latest_version)

    if pull_requests.try(:any?)
      pull_requests = pull_requests.map { |pr| "#{pr[:title]} (#{Formatter.url(pr[:url])})" }.join(", ")
    end

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
  end

  def livecheck_formula(formula)
    ohai "Checking livecheck formula: #{formula}" if Homebrew.args.verbose?

    response = Utils.popen_read(HOMEBREW_BREW_FILE, "livecheck", formula, "--quiet").chomp

    parse_livecheck_response(response)
  end

  def parse_livecheck_response(response)
    # e.g response => aacgain : 7834 ==> 1.8
    output = response.delete(" ").split(/:|==>/)

    # e.g. ["openclonk", "7.0", "8.1"]
    package_name, brew_version, latest_version = output

    {
      name:              package_name,
      formula_version:   brew_version,
      livecheck_version: latest_version,
    }
  end

  def display(outdated_packages)
    puts
    outdated_packages.each do |formula, package_details|
      ohai formula
      puts "Current formula version:  #{package_details[:current_formula_version]}"
      puts "Latest Repology version:  #{package_details[:repology_latest_version]}"
      puts "Latest livecheck version: #{package_details[:livecheck_latest_version] || "Not found"}"
      puts "Open pull requests: #{package_details[:open_pull_requests] || "None"}"
    end
  end
end
