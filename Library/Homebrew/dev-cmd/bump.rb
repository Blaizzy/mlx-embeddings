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

    outdated_packages = Repology.validate_and_format_packages(outdated_repology_packages)
    display(outdated_packages)
  end

  def get_formula_details(formula_name)
    Formula[formula_name]
  rescue
    nil
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
