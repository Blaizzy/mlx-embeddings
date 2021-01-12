# typed: true
# frozen_string_literal: true

require "forwardable"

module RuboCop
  module Cop
    module Cask
      # Do not use the deprecated DSL version syntax in your cask header.
      #
      # @example
      #   # bad
      #   cask :v1 => 'foo' do
      #     ...
      #   end
      #
      #   # good
      #   cask 'foo' do
      #     ...
      #   end
      class NoDslVersion < Base
        extend T::Sig

        extend Forwardable
        extend AutoCorrector
        include CaskHelp

        MESSAGE = "Use `%<preferred>s` instead of `%<current>s`"

        def on_cask(cask_block)
          @cask_header = cask_block.header
          return unless offense?

          offense
        end

        private

        def_delegator :@cask_header, :source_range, :header_range
        def_delegators :@cask_header, :header_str, :preferred_header_str

        def cask_header(method_node)
          RuboCop::Cask::AST::CaskHeader.new(method_node)
        end

        def offense?
          @cask_header.dsl_version?
        end

        def offense
          add_offense(header_range, message: error_msg) do |corrector|
            corrector.replace(header_range, preferred_header_str)
          end
        end

        sig { returns(String) }
        def error_msg
          format(MESSAGE, preferred: preferred_header_str, current: header_str)
        end
      end
    end
  end
end
