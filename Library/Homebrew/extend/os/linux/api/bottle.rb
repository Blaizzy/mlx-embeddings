# typed: false
# frozen_string_literal: true

module Homebrew
  module API
    module Bottle
      class << self
        sig { returns(String) }
        def bottle_api_path
          return generic_bottle_api_path if Homebrew::EnvConfig.force_homebrew_on_linux?

          "bottle-linux"
        end
      end
    end
  end
end
