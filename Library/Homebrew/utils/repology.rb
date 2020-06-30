# frozen_string_literal: true

require "utils/curl"

module RepologyParser
  def call_api(url)
    ohai "- Calling API #{url}" if Homebrew.args.verbose?
    uri = URI(url)
    curl(uri)
  end

  def query_repology_api(last_package_in_response = "")
    url = "https://repology.org/api/v1/projects/#{last_package_in_response}?inrepo=homebrew&outdated=1"

    call_api(url)
  end

  def parse_repology_api
    ohai "Querying outdated packages from Repology"
    page_no = 1
    ohai "\n- Paginating repology api page: #{page_no}"

    outdated_packages = query_repology_api
    last_package_index = outdated_packages.size - 1
    response_size = outdated_packages.size

    while response_size > 1 && page_no <= 15
      page_no += 1
      ohai "\n- Paginating repology api page: #{page_no}"

      last_package_in_response = outdated_packages.keys[last_package_index]
      response = query_repology_api("#{last_package_in_response}/")

      response_size = response.size
      outdated_packages.merge!(response)
      last_package_index = outdated_packages.size - 1
    end

    ohai "\n- #{outdated_packages.size} outdated packages identified"

    outdated_packages
  end

  def validate__repology_packages(outdated_repology_packages)
    ohai "\n- Verify Outdated Repology Packages as Homebrew Formulae"

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

      packages[repology_homebrew_repo["srcname"]] = {
        "repology_latest_version" => latest_version,
      }
    end
    # hash of hashes {'openclonk' => {repology_latest_version => 7.0}, ..}
    packages
  end
end
