# frozen_string_literal: true

module Cask
  class DSL
    class Appcast
      ATTRIBUTES = [
        :configuration,
      ].freeze
      attr_reader :uri, :parameters
      attr_reader(*ATTRIBUTES)

      def initialize(uri, **parameters)
        @uri        = URI(uri)
        @parameters = parameters

        ATTRIBUTES.each do |attribute|
          next unless parameters.key?(attribute)

          instance_variable_set("@#{attribute}", parameters[attribute])
        end
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
