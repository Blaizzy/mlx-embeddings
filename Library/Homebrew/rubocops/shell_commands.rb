# typed: true
# frozen_string_literal: true

require "active_support/core_ext/array/access"
require "rubocops/shared/helper_functions"

module RuboCop
  module Cop
    module Style
      # This cop makes sure that shell command arguments are separated.
      #
      # @api private
      class ShellCommands < Base
        include HelperFunctions
        extend AutoCorrector

        MSG = "Separate `%<method>s` commands into `%<good_args>s`"

        TARGET_METHODS = [
          [nil, :system],
          [nil, :safe_system],
          [nil, :quiet_system],
          [:Utils, :popen_read],
          [:Utils, :safe_popen_read],
          [:Utils, :popen_write],
          [:Utils, :safe_popen_write],
        ].freeze
        RESTRICT_ON_SEND = TARGET_METHODS.map(&:second).uniq.freeze

        SHELL_METACHARACTERS = %w[> < < | ; : & * $ ? : ~ + @ ! ` ( ) [ ]].freeze

        def on_send(node)
          TARGET_METHODS.each do |target_class, target_method|
            next unless node.method_name == target_method

            target_receivers = if target_class.nil?
              [nil, s(:const, nil, :Kernel), s(:const, nil, :Homebrew)]
            else
              [s(:const, nil, target_class)]
            end
            next unless target_receivers.include?(node.receiver)

            first_arg = node.arguments.first
            arg_count = node.arguments.count
            if first_arg&.hash_type? # popen methods allow env hash
              first_arg = node.arguments.second
              arg_count -= 1
            end
            next if first_arg.nil? || arg_count >= 2

            first_arg_str = string_content(first_arg)

            # Only separate when no shell metacharacters are present
            next if SHELL_METACHARACTERS.any? { |meta| first_arg_str.include?(meta) }

            split_args = first_arg_str.shellsplit
            next if split_args.count <= 1

            good_args = split_args.map { |arg| "\"#{arg}\"" }.join(", ")
            method_string = if target_class
              "#{target_class}.#{target_method}"
            else
              target_method.to_s
            end
            add_offense(first_arg, message: format(MSG, method: method_string, good_args: good_args)) do |corrector|
              corrector.replace(first_arg.source_range, good_args)
            end
          end
        end
      end
    end
  end
end
