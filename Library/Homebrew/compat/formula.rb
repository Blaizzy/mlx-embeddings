# frozen_string_literal: true

class Formula
  module Compat
    def installed?
      odeprecated "Formula#installed?",
                  "Formula#latest_version_installed? (or Formula#any_version_installed? )"
      latest_version_installed?
    end

    def prepare_patches
      if respond_to?(:patches)
        active_spec.add_legacy_patches(patches)
        odeprecated "patches", "patch do"
      end

      super
    end
  end

  prepend Compat
end
