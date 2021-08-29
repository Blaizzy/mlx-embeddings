# typed: false
# frozen_string_literal: true

module Homebrew
  module API
    module Analytics
      class << self
        def analytics_api_path
          return generic_analytics_api_path if
            Homebrew::EnvConfig.force_homebrew_on_linux? ||
            Homebrew::EnvConfig.force_homebrew_core_repo_on_linux?

          "analytics-linux"
        end
      end
    end
  end
end
