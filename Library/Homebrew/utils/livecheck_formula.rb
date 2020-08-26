# frozen_string_literal: true

# Helper module for parsing output of `brew livecheck`.
#
# @api private
module LivecheckFormula
  module_function

  def init(formula)
    ohai "Checking livecheck formula: #{formula}" if Homebrew.args.verbose?

    response = Utils.popen_read(HOMEBREW_BREW_FILE, "livecheck", formula, "--quiet").chomp

    parse_livecheck_response(response)
  end

  def parse_livecheck_response(response)
    # e.g response => aacgain : 7834 ==> 1.8
    output = response.delete(" ").split(/:|==>/)

    # e.g. ["openclonk", "7.0", "8.1"]
    package_name, brew_version, latest_version = output

    {
      name:              package_name,
      formula_version:   brew_version,
      livecheck_version: latest_version,
    }
  end
end
