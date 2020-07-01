# frozen_string_literal: true

module Livecheck
  def livecheck_formula_response(name)
    ohai "Checking livecheck formula : #{name}"

    response = Utils.popen_read("brew", "livecheck", name, "--quiet").chomp
    parse_livecheck_response(response)
  end

  def parse_livecheck_response(response)
    output = response.delete(" ").split(/:|==>/)

    # eg: ["burp", "2.2.18", "2.2.18"]
    package_name, brew_version, latest_version = output

    {
      name:              package_name,
      formula_version:   brew_version,
      livecheck_version: latest_version,
    }
  end
end
