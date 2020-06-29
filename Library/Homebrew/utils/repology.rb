require 'net/http'
require 'json'

module Utils
  module RepologyParser
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
  end
end
