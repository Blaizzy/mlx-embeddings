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

    def installed_prefix
      # TODO: deprecate for Homebrew 2.5
      # odeprecated "Formula#installed_prefix",
      #             "Formula#latest_installed_prefix (or Formula#any_installed_prefix)"
      latest_installed_prefix
    end

    # The currently installed version for this formula. Will raise an exception
    # if the formula is not installed.
    # @private
    def installed_version
      # TODO: deprecate for Homebrew 2.5
      # odeprecated "Formula#installed_version"
      Keg.new(latest_installed_prefix).version
    end

    def opt_or_installed_prefix_keg
      # TODO: deprecate for Homebrew 2.5
      # odeprecated "Formula#opt_or_installed_prefix_keg", "Formula#any_installed_keg"
      any_installed_keg
    end
  end

  prepend Compat
end
