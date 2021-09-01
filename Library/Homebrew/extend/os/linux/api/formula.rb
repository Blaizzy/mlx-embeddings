# typed: false
# frozen_string_literal: true

module Homebrew
  module API
    module Formula
      class << self
        def formula_api_path
          return generic_formula_api_path if
            Homebrew::EnvConfig.force_homebrew_on_linux? ||
            Homebrew::EnvConfig.force_homebrew_core_repo_on_linux?

          "formula-linux"
        end
      end
    end
  end
end
