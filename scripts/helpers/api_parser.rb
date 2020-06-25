require 'net/http'
require 'json'

require_relative 'brew_commands'
require_relative 'homebrew_formula'

class ApiParser
  def call_api(url)
    puts "- Calling API #{url}"
    uri = URI(url)
    response = Net::HTTP.get(uri)

    puts "- Parsing response"
    JSON.parse(response)
  end

  def query_repology_api(last_package_in_response = '')
    url = 'https://repology.org/api/v1/projects/' + last_package_in_response + '?inrepo=homebrew&outdated=1'

    self.call_api(url)
  end

  def parse_repology_api()
    puts "\n-------- Query outdated packages from Repology --------"
    page_no = 1
    puts "\n- Paginating repology api page: #{page_no}"

    outdated_packages = self.query_repology_api('')
    last_pacakge_index = outdated_packages.size - 1
    response_size = outdated_packages.size

    while response_size > 1  do
      page_no += 1
      puts "\n- Paginating repology api page: #{page_no}"

      last_package_in_response = outdated_packages.keys[last_pacakge_index]
      response = self.query_repology_api("#{last_package_in_response}/")

      response_size = response.size
      outdated_packages.merge!(response)
      last_pacakge_index = outdated_packages.size - 1
    end

    puts "\n- #{outdated_packages.size} outdated pacakges identified by repology"
    outdated_packages
  end

  def query_homebrew
    puts "\n-------- Get Homebrew Formulas --------"
    self.call_api('https://formulae.brew.sh/api/formula.json')
  end

  def parse_homebrew_formulas()
    formulas = self.query_homebrew()
    parsed_homebrew_formulas = {}

    formulas.each do |formula|
      parsed_homebrew_formulas[formula['name']] = {
        "fullname" => formula["full_name"],
        "oldname" => formula["oldname"],
        "version" => formula["versions"]['stable'],
        "download_url" => formula["urls"]['stable']['url'],
      }
    end

    parsed_homebrew_formulas
  end

  def validate_packages(outdated_repology_packages, brew_formulas)
    puts "\n-------- Verify Outdated Repology packages as Homebrew Formulas --------"
    packages = {}

    outdated_repology_packages.each do |package_name, repo_using_package|
      # Identify homebrew repo
      repology_homebrew_repo = repo_using_package.select { |repo| repo['repo'] == 'homebrew' }[0]
      next if repology_homebrew_repo.empty?

      latest_version = nil

      # Identify latest version amongst repos
      repo_using_package.each do |repo|
        latest_version = repo['version'] if repo['status'] == 'newest'
      end

      repology_homebrew_repo['latest_version'] = latest_version if latest_version
      homebrew_package_details = brew_formulas[repology_homebrew_repo['srcname']]
      
      # Format package
      packages[repology_homebrew_repo['srcname']] = format_package(homebrew_package_details, repology_homebrew_repo)
    end

    packages
  end


  def format_package(homebrew_details, repology_details)
    puts "- Formatting package: #{repology_details['srcname']}"

    homebrew_formula = HomebrewFormula.new
    new_download_url = homebrew_formula.generate_new_download_url(homebrew_details['download_url'], homebrew_details['version'], repology_details['latest_version'])

    brew_commands = BrewCommands.new
    livecheck_response = brew_commands.livecheck_check_formula(repology_details['srcname'])
    has_open_pr = brew_commands.check_for_open_pr(repology_details['srcname'], new_download_url)
   
    formatted_package = {
      'fullname'=> homebrew_details['fullname'],
      'repology_version' => repology_details['latest_version'],
      'homebrew_version' => homebrew_details['version'],
      'livecheck_latest_version' => livecheck_response['livecheck_latest_version'],
      'current_download_url' => homebrew_details['download_url'],
      'latest_download_url' => new_download_url,
      'repology_latest_version' => repology_details['latest_version'],
      'has_open_pr' => has_open_pr
    } 

    formatted_package
  end

  def display_version_data(outdated_packages)
    puts "==============Formatted outdated packages============\n"

    outdated_packages.each do |package_name, package_details|
      puts ""
      puts "Package: #{package_name}"
      puts "Brew current: #{package_details['homebrew_version']}"
      puts "Repology latest: #{package_details['repology_version']}"
      puts "Livecheck latest: #{package_details['livecheck_latest_version']}"
      puts "Has Open PR?: #{package_details['has_open_pr']}"
    end
  end

end
