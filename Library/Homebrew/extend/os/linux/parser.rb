# typed: true
# frozen_string_literal: true

module Homebrew
  module CLI
    class Parser
      undef set_default_options
      undef validate_options

      def set_default_options
        @args["formula?"] = true if @args.respond_to?(:formula?)
      end

      def validate_options
        return unless @args.respond_to?(:cask?)
        return unless @args.cask?

        # NOTE: We don't raise a UsageError here because
        # we don't want to print the help page.
        raise "Invalid usage: Casks are not supported on Linux"
      end
    end
  end
end
