require "open3"

class BrewCommands
 
  def livecheck_check_formula(formula_name)
    puts "- livecheck formula : #{formula_name}"
    command_args = [
      "brew",
      "livecheck",
      formula_name,
      "--quiet",
    ]

    response = Open3.capture2e(*command_args)
    self.parse_livecheck_response(response)
  end

  def parse_livecheck_response(livecheck_output)
    livecheck_output = livecheck_output.first.gsub(' ', '').split(/:|==>|\n/)

    # eg: ["burp", "2.2.18", "2.2.18"]
    package_name, brew_version, latest_version = livecheck_output
  
    {'name' => package_name, 'current_brew_version' => brew_version, 'livecheck_latest_version' => latest_version}
  end

  def bump_formula_pr(formula_name, url)
    command_args = [
      "brew",
      "bump-formula-pr",
      "--no-browse",
      "--dry-run",
      formula_name,
      "--url=#{url}",
    ]

    response = Open3.capture2e(*command_args)
    self.parse_formula_bump_response(response)
  end

  def parse_formula_bump_response(formula_bump_response)
    response, status  = formula_bump_response
    response    
  end

  def check_for_open_pr(formula_name, download_url)
    puts "- Checking for open PRs for formula : #{formula_name}"

    response =  bump_formula_pr(formula_name, download_url)

    return true if !response.include? 'Error: These open pull requests may be duplicates'
    false
  end  

end