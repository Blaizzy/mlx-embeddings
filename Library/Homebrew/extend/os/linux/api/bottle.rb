# typed: false
# frozen_string_literal: true

module Homebrew
  module API
    module Bottle
      class << self
        def bottle_api_path
          return generic_bottle_api_path if
            Homebrew::EnvConfig.force_homebrew_on_linux? ||
            Homebrew::EnvConfig.force_homebrew_core_repo_on_linux?

          "bottle-linux"
        end
      end
    end
  end
end
