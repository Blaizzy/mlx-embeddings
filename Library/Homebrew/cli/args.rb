# typed: true
# frozen_string_literal: true

require "ostruct"

module Homebrew
  module CLI
    class Args < OpenStruct
      extend T::Sig

      attr_reader :options_only, :flags_only

      # undefine tap to allow --tap argument
      undef tap

      sig { void }
      def initialize
        require "cli/named_args"

        super()

        @processed_options = []
        @options_only = []
        @flags_only = []
        @cask_options = false

        # Can set these because they will be overwritten by freeze_named_args!
        # (whereas other values below will only be overwritten if passed).
        self[:named] = NamedArgs.new(parent: self)
        self[:remaining] = []
      end

      def freeze_remaining_args!(remaining_args)
        self[:remaining] = remaining_args.freeze
      end

      def freeze_named_args!(named_args, cask_options:)
        self[:named] = NamedArgs.new(
          *named_args.freeze,
          override_spec: spec(nil),
          force_bottle:  self[:force_bottle?],
          flags:         flags_only,
          cask_options:  cask_options,
          parent:        self,
        )
      end

      def freeze_processed_options!(processed_options)
        # Reset cache values reliant on processed_options
        @cli_args = nil

        @processed_options += processed_options
        @processed_options.freeze

        @options_only = cli_args.select { |a| a.start_with?("-") }.freeze
        @flags_only = cli_args.select { |a| a.start_with?("--") }.freeze
      end

      sig { returns(NamedArgs) }
      def named
        require "formula"
        self[:named]
      end

      def no_named?
        named.blank?
      end

      def build_from_source_formulae
        if build_from_source? || self[:HEAD?] || self[:build_bottle?]
          named.to_formulae.map(&:full_name)
        else
          []
        end
      end

      def include_test_formulae
        if include_test?
          named.to_formulae.map(&:full_name)
        else
          []
        end
      end

      def value(name)
        arg_prefix = "--#{name}="
        flag_with_value = flags_only.find { |arg| arg.start_with?(arg_prefix) }
        return unless flag_with_value

        flag_with_value.delete_prefix(arg_prefix)
      end

      sig { returns(Context::ContextStruct) }
      def context
        Context::ContextStruct.new(debug: debug?, quiet: quiet?, verbose: verbose?)
      end

      def only_formula_or_cask
        return :formula if formula? && !cask?
        return :cask if cask? && !formula?
      end

      private

      def option_to_name(option)
        option.sub(/\A--?/, "")
              .tr("-", "_")
      end

      def cli_args
        return @cli_args if @cli_args

        @cli_args = []
        @processed_options.each do |short, long|
          option = long || short
          switch = "#{option_to_name(option)}?".to_sym
          flag = option_to_name(option).to_sym
          if @table[switch] == true || @table[flag] == true
            @cli_args << option
          elsif @table[flag].instance_of? String
            @cli_args << "#{option}=#{@table[flag]}"
          elsif @table[flag].instance_of? Array
            @cli_args << "#{option}=#{@table[flag].join(",")}"
          end
        end
        @cli_args.freeze
      end

      def spec(default = :stable)
        if self[:HEAD?]
          :head
        else
          default
        end
      end

      def respond_to_missing?(*)
        !frozen?
      end

      def method_missing(method_name, *args)
        return_value = super

        # Once we are frozen, verify any arg method calls are already defined in the table.
        # The default OpenStruct behaviour is to return nil for anything unknown.
        if frozen? && args.empty? && !@table.key?(method_name)
          raise NoMethodError, "CLI arg for `#{method_name}` is not declared for this command"
        end

        return_value
      end
    end
  end
end
