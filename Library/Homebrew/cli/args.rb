# frozen_string_literal: true

require "ostruct"

module Homebrew
  module CLI
    class Args < OpenStruct
      attr_accessor :processed_options
      # undefine tap to allow --tap argument
      undef tap

      def initialize(argv:)
        super
        @argv = argv
        @processed_options = []
      end

      def option_to_name(option)
        option.sub(/\A--?/, "")
              .tr("-", "_")
      end

      def cli_args
        return @cli_args unless @cli_args.nil?

        @cli_args = []
        processed_options.each do |short, long|
          option = long || short
          switch = "#{option_to_name(option)}?".to_sym
          flag = option_to_name(option).to_sym
          if @table[switch].instance_of? TrueClass
            @cli_args << option
          elsif @table[flag].instance_of? TrueClass
            @cli_args << option
          elsif @table[flag].instance_of? String
            @cli_args << option + "=" + @table[flag]
          elsif @table[flag].instance_of? Array
            @cli_args << option + "=" + @table[flag].join(",")
          end
        end
        @cli_args
      end

      def options_only
        @options_only ||= cli_args.select { |arg| arg.start_with?("-") }
      end

      def flags_only
        @flags_only ||= cli_args.select { |arg| arg.start_with?("--") }
      end
    end
  end
end
