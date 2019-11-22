# frozen_string_literal: true

module Utils
  module Analytics
    class << self
      def formula_path
        return generic_formula_path if ENV["HOMEBREW_FORCE_HOMEBREW_ON_LINUX"]

        "formula-linux"
      end

      def analytics_path
        return generic_analytics_path if ENV["HOMEBREW_FORCE_HOMEBREW_ON_LINUX"]

        "analytics-linux"
      end
    end
  end
end
