# typed: true
# frozen_string_literal: true

class Formula
  module Compat
    def installed_prefix
      odisabled "Formula#installed_prefix",
                "Formula#latest_installed_prefix (or Formula#any_installed_prefix)"
    end

    # The currently installed version for this formula. Will raise an exception
    # if the formula is not installed.
    # @private
    def installed_version
      odisabled "Formula#installed_version"
    end

    def opt_or_installed_prefix_keg
      odisabled "Formula#opt_or_installed_prefix_keg", "Formula#any_installed_keg"
    end
  end

  prepend Compat
end
