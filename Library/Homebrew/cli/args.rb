# frozen_string_literal: true

require "cli/named_args"
require "ostruct"

module Homebrew
  module CLI
    class Args < OpenStruct
      attr_reader :options_only, :flags_only

      # undefine tap to allow --tap argument
      undef tap

      def initialize
        super()

        @processed_options = []
        @options_only = []
        @flags_only = []

        # Can set these because they will be overwritten by freeze_named_args!
        # (whereas other values below will only be overwritten if passed).
        self[:named_args] = NamedArgs.new
        self[:remaining] = []
      end

      def freeze_remaining_args!(remaining_args)
        self[:remaining] = remaining_args.freeze
      end

      def freeze_named_args!(named_args)
        self[:named_args] = NamedArgs.new(
          *named_args.freeze,
          override_spec: spec(nil),
          force_bottle:  force_bottle?,
          flags:         flags_only,
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

      def named
        named_args || NamedArgs.new
      end

      def no_named?
        named.blank?
      end

      def formulae
        odeprecated "args.formulae", "args.named.to_formulae"
        named.to_formulae
      end

      def formulae_and_casks
        odeprecated "args.formulae_and_casks", "args.named.to_formulae_and_casks"
        named.to_formulae_and_casks
      end

      def resolved_formulae
        odeprecated "args.resolved_formulae", "args.named.to_resolved_formulae"
        named.to_resolved_formulae
      end

      def resolved_formulae_casks
        odeprecated "args.resolved_formulae_casks", "args.named.to_resolved_formulae_to_casks"
        named.to_resolved_formulae_to_casks
      end

      def formulae_paths
        odeprecated "args.formulae_paths", "args.named.to_formulae_paths"
        named.to_formulae_paths
      end

      def casks
        odeprecated "args.casks", "args.named.homebrew_tap_cask_names"
        named.homebrew_tap_cask_names
      end

      def loaded_casks
        odeprecated "args.loaded_casks", "args.named.to_cask"
        named.to_casks
      end

      def kegs
        odeprecated "args.kegs", "args.named.to_kegs"
        named.to_kegs
      end

      def kegs_casks
        odeprecated "args.kegs", "args.named.to_kegs_to_casks"
        named.to_kegs_to_casks
      end

      def build_stable?
        !HEAD?
      end

      def build_from_source_formulae
        if build_from_source? || build_bottle?
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

      def context
        Context::ContextStruct.new(debug: debug?, quiet: quiet?, verbose: verbose?)
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
        if HEAD?
          :head
        else
          default
        end
      end
    end
  end
end
