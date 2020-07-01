require_relative "helpers/api_parser"

api_parser = ApiParser.new

outdated_repology_packages = api_parser.parse_repology_api()
brew_formulas = api_parser.parse_homebrew_formulas()

formatted_outdated_packages = api_parser.validate_packages(outdated_repology_packages, brew_formulas)

api_parser.display_version_data(formatted_outdated_packages)
