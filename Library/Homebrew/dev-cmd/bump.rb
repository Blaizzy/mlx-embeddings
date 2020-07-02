# frozen_string_literal: true

require "cli/parser"
require "utils/popen"

module Homebrew
  module_function

  def bump_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `bump`

        Display out-of-date brew formulae, the latest version available, and whether a pull request has been opened.
      EOS
    end
  end

  def bump
    bump_args.parse

    outdated_repology_packages = RepologyParser.parse_api_response
    outdated_packages = RepologyParser.validate_and_format_packages(outdated_repology_packages)

    display(outdated_packages)
  end

  def display(outdated_packages)
    ohai "Outdated Formulae"

    outdated_packages.each do |formula, package_details|
      puts ""
      puts "Formula: #{formula}"
      puts "Current formula version: #{package_details["current_formula_version"]}"
      puts "Latest repology version: #{package_details["repology_latest_version"]}"
      puts "Latest livecheck version: #{package_details["livecheck_latest_version"]}"
      puts "Open pull requests: #{package_details["open_pull_requests"]}"
    end
  end
end
