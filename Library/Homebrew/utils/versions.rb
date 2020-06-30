# frozen_string_literal: true

require "open3"
require "formula"

module Versions
  def current_formula_version(formula_name)
    Formula[formula_name].version.to_s.to_f
  end

  def bump_formula_pr(formula_name, url)
    command_args = [
      "brew",
      "bump-formula-pr",
      "--no-browse",
      "--dry-run",
      formula_name,
      "--url=#{url}",
    ]

    response = Open3.capture2e(*command_args)
    parse_formula_bump_response(response)
  end

  def parse_formula_bump_response(response)
    response, status  = formula_bump_response
    response
  end

  def check_for_open_pr(formula_name, download_url)
    puts "- Checking for open PRs for formula : #{formula_name}"

    response = bump_formula_pr(formula_name, download_url)
    !response.include? 'Error: These open pull requests may be duplicates'
  end
end
