# frozen_string_literal: true

module Cask
  class DSL
    class Appcast
      attr_reader :uri, :parameters, :configuration

      def initialize(uri, **parameters)
        @uri        = URI(uri)
        @parameters = parameters
        @configuration = parameters[:configuration] if parameters.key?(:configuration)
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
