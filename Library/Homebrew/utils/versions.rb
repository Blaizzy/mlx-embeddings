# frozen_string_literal: true

require "formula"

module Versions
  def current_formula_version(formula_name)
    Formula[formula_name].version.to_s.to_f
  end

  def bump_formula_pr(formula_name, url)
    Utils.popen_read("brew", "bump-formula-pr", "--no-browse", "--dry-run",
                     formula_name, "--url=#{url}").chomp
  end

  def check_for_open_pr(formula_name, download_url)
    ohai "- Checking for open PRs for formula : #{formula_name}"

    response = bump_formula_pr(formula_name, download_url)
    !response.include? "Error: These open pull requests may be duplicates"
  end
end
