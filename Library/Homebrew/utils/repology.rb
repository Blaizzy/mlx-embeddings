# typed: false
# frozen_string_literal: true

require "utils/curl"

# Repology API client.
#
# @api private
module Repology
  HOMEBREW_CORE = "homebrew"
  HOMEBREW_CASK = "homebrew_casks"

  module_function

  MAX_PAGINATION = 15
  private_constant :MAX_PAGINATION

  def query_api(last_package_in_response = "", repository:)
    last_package_in_response += "/" if last_package_in_response.present?
    url = "https://repology.org/api/v1/projects/#{last_package_in_response}?inrepo=#{repository}&outdated=1"

    output, _errors, _status = curl_output(url.to_s)
    JSON.parse(output)
  end

  def single_package_query(name, repository:)
    url = "https://repology.org/tools/project-by?repo=#{repository}&" \
          "name_type=srcname&target_page=api_v1_project&name=#{name}"

    output, _errors, _status = curl_output("--location", url.to_s)

    begin
      data = JSON.parse(output)
      { name => data }
    rescue
      nil
    end
  end

  def parse_api_response(limit = nil, repository:)
    package_term = case repository
    when HOMEBREW_CORE
      "formula"
    when HOMEBREW_CASK
      "cask"
    else
      "package"
    end

    ohai "Querying outdated #{package_term.pluralize} from Repology"

    page_no = 1
    outdated_packages = {}
    last_package = ""

    while page_no <= MAX_PAGINATION
      odebug "Paginating Repology API page: #{page_no}"

      response = query_api(last_package, repository: repository)
      outdated_packages.merge!(response)
      last_package = response.keys.last

      page_no += 1
      break if limit && outdated_packages.size >= limit || response.size <= 1
    end

    puts "#{outdated_packages.size} outdated #{package_term.pluralize(outdated_packages.size)} found"
    puts

    outdated_packages
  end

  def latest_version(repositories)
    # The status is "unique" when the package is present only in Homebrew, so
    # Repology has no way of knowing if the package is up-to-date.
    is_unique = repositories.find do |repo|
      repo["status"] == "unique"
    end.present?

    return "present only in Homebrew" if is_unique

    latest_version = repositories.find do |repo|
      repo["status"] == "newest"
    end

    # Repology cannot identify "newest" versions for packages without a version
    # scheme
    return "no latest version" if latest_version.blank?

    latest_version["version"]
  end
end
