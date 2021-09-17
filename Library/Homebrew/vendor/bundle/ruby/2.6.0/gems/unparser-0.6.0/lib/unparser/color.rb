# frozen_string_literal: true

module Unparser
  # Class to colorize strings
  class Color
    include Adamantium, Concord.new(:code)

    # Format text with color
    #
    # @param [String] text
    #
    # @return [String]
    def format(text)
      "\e[#{code}m#{text}\e[0m"
    end

    NONE = Class.new(self) do

      # Format null color
      #
      # @param [String] text
      #
      # @return [String]
      #   the argument string
      def format(text)
        text
      end

    private

      def initialize; end

    end.new

    RED   = Color.new(31)
    GREEN = Color.new(32)

  end # Color
end # Unparser
