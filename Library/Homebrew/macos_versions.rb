# typed: true
# frozen_string_literal: true

# Helper functions for querying operating system information.
#
# @api private
module MacOSVersions
  # TODO: when removing symbols here, ensure that they are added to
  # DEPRECATED_MACOS_VERSIONS in MacOSRequirement.
  SYMBOLS = {
    ventura:     "13",
    monterey:    "12",
    big_sur:     "11",
    catalina:    "10.15",
    mojave:      "10.14",
    high_sierra: "10.13",
    sierra:      "10.12",
    el_capitan:  "10.11",
  }.freeze
end
