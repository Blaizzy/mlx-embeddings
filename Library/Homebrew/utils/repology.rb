# frozen_string_literal: true

require "utils/curl"
require "utils/versions"
require "formula_info"

module RepologyParser
  module_function

  MAX_PAGINATION = 15

  def query_api(last_package_in_response = "")
    last_package_in_response += "/" unless last_package_in_response.empty?

    url = "https://repology.org/api/v1/projects/#{last_package_in_response}?inrepo=homebrew&outdated=1"
    ohai "Calling API #{url}" if Homebrew.args.verbose?

    output, _errors, _status = curl_output(url.to_s)
    JSON.parse(output)
  end

  def parse_api_response
    ohai "Querying outdated packages from Repology"

    outdated_packages = query_api
    last_package_index = outdated_packages.size - 1
    response_size = outdated_packages.size
    page_no = 1

    while response_size > 1 && page_no <= MAX_PAGINATION
      ohai "Paginating Repology api page: #{page_no}" if Homebrew.args.verbose?

      last_package_in_response = outdated_packages.keys[last_package_index]
      response = query_api(last_package_in_response)

      response_size = response.size
      outdated_packages.merge!(response)
      last_package_index = outdated_packages.size - 1
      page_no += 1
    end

    ohai "#{outdated_packages.size} outdated packages identified"

    outdated_packages
  end

  def validate_and_format_packages(outdated_repology_packages)
    ohai "Verifying outdated Repology packages as Homebrew Formulae"

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
    # hash of hashes {"aacgain"=>{"repology_latest_version"=>"1.9", "current_formula_version"=>"1.8"}, ...}
    packages
  end

  def format_package(package_name, latest_version)
    current_version = Versions.current_formula_version(package_name)
    livecheck_response = Versions.livecheck_formula(package_name)
    pull_requests = Versions.check_for_duplicate_pull_requests(package_name, latest_version)

    {
      repoology_latest_version: latest_version,
      current_formula_version:  current_version.to_s,
      livecheck_latest_version: livecheck_response[:livecheck_version],
      open_pull_requests:       pull_requests,
    }
  end
end
