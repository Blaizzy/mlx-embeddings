# typed: true
# frozen_string_literal: true

module Cask
  class DSL
    # Class corresponding to the `appcast` stanza.
    #
    # @api private
    class Appcast
      attr_reader :uri, :parameters, :must_contain

      def initialize(uri, **parameters)
        @uri        = URI(uri)
        @parameters = parameters
        @must_contain = parameters[:must_contain] if parameters.key?(:must_contain)
      end

      def to_yaml
        [uri, parameters].to_yaml
      end

      def to_s
        uri.to_s
      end
    end
  end
end
