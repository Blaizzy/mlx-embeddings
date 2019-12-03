# frozen_string_literal: true

class Formula
  module Compat
    def installed?
      # odeprecated "Formula#installed?",
      #             "Formula#latest_version_installed? (or Formula#any_version_installed? )"
      latest_version_installed?
    end
  end

  prepend Compat
end
