# typed: true
# frozen_string_literal: true

require "cli/parser"

module Homebrew
  extend T::Sig

  module_function

  sig { returns(CLI::Parser) }
  def bump_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `bump` [<options>] [<formula>]

        Display out-of-date brew formulae and the latest version available.
        Also displays whether a pull request has been opened with the URL.
      EOS
      flag   "--limit=",
             description: "Limit number of package results returned."
      switch :verbose
      switch :debug

      named_args :formula
    end
  end

  def bump
    args = bump_args.parse

    requested_formulae = args.named.to_formulae.map(&:name) if args.named.to_formulae.present?

    requested_limit = args.limit.to_i if args.limit.present?

    repology_data = if requested_formulae
      response = {}
      requested_formulae.each do |formula|
        raise FormulaUnavailableError, formula unless validate_formula(formula)

        package_data = Repology.single_package_query(formula)
        response[package_data.keys.first] = package_data.values.first if package_data
      end

      response
    else
      Repology.parse_api_response(requested_limit)
    end

    validated_formulae = {}

    validated_formulae = Repology.validate_and_format_packages(repology_data, requested_limit) if repology_data

    if requested_formulae
      repology_excluded_formulae = requested_formulae.reject do |formula|
        repology_data[formula]
      end

      formulae = {}
      repology_excluded_formulae.each do |formula|
        formulae[formula] = Repology.format_package(formula, nil)
      end

      formulae.each { |formula, data| validated_formulae[formula] = data }
    end

    display(validated_formulae)
  end

  def validate_formula(formula_name)
    Formula[formula_name]
  rescue
    nil
  end

  def up_to_date?(package)
    package &&
      package[:current_formula_version] == package[:repology_latest_version] &&
      package[:current_formula_version] == package[:livecheck_latest_version]
  end

  def display(formulae)
    formulae.each do |formula, package_details|
      title = (up_to_date?(package_details) ? "#{formula} is up to date!" : formula).to_s
      ohai title
      puts "Current formula version:  #{package_details[:current_formula_version]}"
      puts "Latest Repology version:  #{package_details[:repology_latest_version]}"
      puts "Latest livecheck version: #{package_details[:livecheck_latest_version]}"
      puts "Open pull requests:       #{package_details[:open_pull_requests]}"
    end
  end
end
