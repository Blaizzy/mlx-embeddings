# typed: true
# frozen_string_literal: true

module Homebrew
  module CLI
    class Parser
      undef validate_options

      def validate_options
        return unless @args.respond_to?(:cask?)
        return unless @args.cask?

        msg = "Casks are not supported on Linux"
        raise UsageError, msg unless Homebrew::EnvConfig.developer?

        opoo msg unless @args.quiet?
      end
    end
  end
end
