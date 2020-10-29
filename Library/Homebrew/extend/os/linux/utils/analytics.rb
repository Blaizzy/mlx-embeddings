# typed: strict
# frozen_string_literal: true

module Utils
  module Analytics
    class << self
      extend T::Sig
      sig { returns(String) }
      def formula_path
        return generic_formula_path if Homebrew::EnvConfig.force_homebrew_on_linux?

        "formula-linux"
      end

      sig { returns(String) }
      def analytics_path
        return generic_analytics_path if Homebrew::EnvConfig.force_homebrew_on_linux?

        "analytics-linux"
      end
    end
  end
end
