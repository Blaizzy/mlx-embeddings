# frozen_string_literal: true

module Versions
  module_function

  def current_formula_version(formula_name)
    Formula[formula_name].version.to_s.to_f
  rescue
    nil
  end

  def livecheck_formula(formula)
    ohai "Checking livecheck formula : #{formula}" if Homebrew.args.verbose?

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

  def fetch_pull_requests(query, tap_full_name, state: nil)
    GitHub.issues_for_formula(query, tap_full_name: tap_full_name, state: state).select do |pr|
      pr["html_url"].include?("/pull/") &&
        /(^|\s)#{Regexp.quote(query)}(:|\s|$)/i =~ pr["title"]
    end
  rescue GitHub::RateLimitExceededError => e
    opoo e.message
    []
  end

  def check_for_duplicate_pull_requests(formula, version)
    formula = Formula[formula]
    tap_full_name = formula.tap&.full_name

    # check for open requests
    pull_requests = fetch_pull_requests(formula.name, tap_full_name, state: "open")

    # if we haven't already found open requests, try for an exact match across all requests
    pull_requests = fetch_pull_requests("#{formula.name} #{version}", tap_full_name) if pull_requests.blank?
    return if pull_requests.blank?

    pull_requests.map { |pr| { title: pr["title"], url: pr["html_url"] } }
  end
end
