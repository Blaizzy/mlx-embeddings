# typed: true
# frozen_string_literal: true

require "cli/parser"
require "livecheck/livecheck"
require "livecheck/skip_conditions"

module Homebrew
  extend T::Sig

  module_function

  sig { returns(CLI::Parser) }
  def bump_args
    Homebrew::CLI::Parser.new do
      description <<~EOS
        Display out-of-date brew formulae and the latest version available.
        Also displays whether a pull request has been opened with the URL.
      EOS
      flag   "--limit=",
             description: "Limit number of package results returned."

      named_args :formula
    end
  end

  def bump
    args = bump_args.parse

    requested_formulae = args.named.to_formulae.presence

    requested_limit = args.limit.to_i if args.limit.present?

    if requested_formulae
      Livecheck.load_other_tap_strategies(requested_formulae)

      requested_formulae.each_with_index do |formula, i|
        puts if i.positive?

        if formula.head_only?
          ohai formula.name
          puts "Formula is HEAD-only."
          next
        end

        current_version = formula.stable.version.to_s

        package_data = Repology.single_package_query(formula.name)
        repology_latest = if package_data.present?
          Repology.latest_version(package_data.values.first)
        else
          "not found"
        end

        livecheck_latest = livecheck_result(formula)
        pull_requests = retrieve_pull_requests(formula)
        display(formula, current_version, repology_latest, livecheck_latest, pull_requests)
      end
    else
      outdated_packages = Repology.parse_api_response(requested_limit)
      outdated_packages.each_with_index do |(_name, repositories), i|
        puts if i.positive?

        homebrew_repo = repositories.find do |repo|
          repo["repo"] == "homebrew"
        end

        next if homebrew_repo.blank?

        formula = begin
          Formula[homebrew_repo["srcname"]]
        rescue
          next
        end

        current_version = formula.stable.version.to_s
        repology_latest = Repology.latest_version(repositories)
        livecheck_latest = livecheck_result(formula)
        pull_requests = retrieve_pull_requests(formula)
        display(formula, current_version, repology_latest, livecheck_latest, pull_requests)

        break if requested_limit && i >= requested_limit
      end
    end
  end

  def livecheck_result(formula)
    skip_result = Livecheck::SkipConditions.skip_information(formula)
    if skip_result.present?
      return "#{skip_result[:status]}#{" - #{skip_result[:messages].join(", ")}" if skip_result[:messages].present?}"
    end

    version_info = Livecheck.latest_version(
      formula,
      json: true, full_name: false, verbose: false, debug: false,
    )
    latest = version_info[:latest] if version_info.present?

    return "not found" if latest.blank?

    latest.to_s
  end

  def retrieve_pull_requests(formula)
    pull_requests = GitHub.fetch_pull_requests(formula.name, formula.tap&.full_name, state: "open")
    if pull_requests.try(:any?)
      pull_requests = pull_requests.map { |pr| "#{pr["title"]} (#{Formatter.url(pr["html_url"])})" }.join(", ")
    end

    return "none" if pull_requests.blank?

    pull_requests
  end

  def up_to_date?(current_version, repology_latest, livecheck_latest)
    current_version == repology_latest &&
      current_version == livecheck_latest
  end

  def display(formula, current_version, repology_latest, livecheck_latest, pull_requests)
    title = if up_to_date?(current_version, repology_latest, livecheck_latest)
      "#{formula} is up to date!"
    else
      formula.name
    end

    ohai title
    puts "Current formula version:  #{current_version}"
    puts "Latest Repology version:  #{repology_latest}"
    puts "Latest livecheck version: #{livecheck_latest}"
    puts "Open pull requests:       #{pull_requests}"
  end
end
