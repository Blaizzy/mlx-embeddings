# frozen_string_literal: true

module Unparser
  class Anima
    # Abstract base class for anima errors
    class Error < RuntimeError
      FORMAT = '%s attributes missing: %s, unknown: %s'.freeze
      private_constant(*constants(false))

      # Initialize object
      #
      # @param [Class] klass
      #   the class being initialized
      # @param [Enumerable<Symbol>] missing
      # @param [Enumerable<Symbol>] unknown
      #
      # @return [undefined]
      def initialize(klass, missing, unknown)
        super(format(FORMAT, klass, missing, unknown))
      end
    end # Error
  end # Anima
end # Unparser
