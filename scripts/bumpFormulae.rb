require_relative "helpers/parsed_file"
require_relative "helpers/brew_commands.rb"

brew_commands = BrewCommands.new

parsed_file = ParsedFile.new
outdated_pckgs_to_update = parsed_file.get_latest_file("data/outdated_pckgs_to_update")

File.foreach(outdated_pckgs_to_update) do |line|
  line_hash = eval(line)
  puts "\n bumping package: #{line_hash['name']} formula"

  begin
    bump_pr_response, bump_pr_status = brew_commands.bump_formula_pr(line_hash["name"], line_hash["download_url"], line_hash["checksum"])
    puts "#{bump_pr_response}"
  rescue
    puts "- An error occured whilst bumping package #{line_hash["name"]} \n"
    return
  end
end
