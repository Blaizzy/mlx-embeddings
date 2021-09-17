# frozen_string_literal: true

module Unparser
  class Anima
    # An attribute
    class Attribute
      include Adamantium, Equalizer.new(:name)

      # Initialize attribute
      #
      # @param [Symbol] name
      def initialize(name)
        @name = name
        @instance_variable_name = :"@#{name}"
      end

      # Return attribute name
      #
      # @return [Symbol]
      attr_reader :name

      # Return instance variable name
      #
      # @return [Symbol]
      attr_reader :instance_variable_name

      # Load attribute
      #
      # @param [Object] object
      # @param [Hash] attributes
      #
      # @return [self]
      def load(object, attributes)
        set(object, attributes.fetch(name))
      end

      # Get attribute value from object
      #
      # @param [Object] object
      #
      # @return [Object]
      def get(object)
        object.public_send(name)
      end

      # Set attribute value in object
      #
      # @param [Object] object
      # @param [Object] value
      #
      # @return [self]
      def set(object, value)
        object.instance_variable_set(instance_variable_name, value)

        self
      end
    end # Attribute
  end # Anima
end # Unparser
