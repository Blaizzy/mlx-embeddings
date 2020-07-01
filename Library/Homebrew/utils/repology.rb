# frozen_string_literal: true

require "utils/curl"
require "formula_info"

module RepologyParser
  module_function

  def query_api(last_package_in_response = "")
    url = "https://repology.org/api/v1/projects/#{last_package_in_response}?inrepo=homebrew&outdated=1"
    ohai "Calling API #{url}" if Homebrew.args.verbose?

    output, errors, status = curl_output(url.to_s)
    output = JSON.parse(output)
  end

  def parse_api_response()
    ohai "Querying outdated packages from Repology"
    page_no = 1
    ohai "Paginating repology api page: #{page_no}" if Homebrew.args.verbose?

    outdated_packages = query_api()
    last_pacakge_index = outdated_packages.size - 1
    response_size = outdated_packages.size
    page_limit = 15

    while response_size > 1 && page_no <= page_limit
      page_no += 1
      ohai "Paginating repology api page: #{page_no}" if Homebrew.args.verbose?

      last_package_in_response = outdated_packages.keys[last_pacakge_index]
      response = query_api("#{last_package_in_response}/")

      response_size = response.size
      outdated_packages.merge!(response)
      last_pacakge_index = outdated_packages.size - 1
    end

    ohai "#{outdated_packages.size} outdated packages identified"

    outdated_packages
  end

  def validate__packages(outdated_repology_packages)
    ohai "Verifying outdated repology packages as Homebrew Formulae"

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

      info = FormulaInfo.lookup(repology_homebrew_repo["srcname"])
      next unless info
      current_version = info.pkg_version
      
      packages[repology_homebrew_repo["srcname"]] = {
        "repology_latest_version" => latest_version,
        "current_formula_version" => current_version.to_s
      }
      puts packages 
    end
    # hash of hashes {"aacgain"=>{"repology_latest_version"=>"1.9", "current_formula_version"=>"1.8"}, ...}
    packages
  end
end
