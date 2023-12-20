# typed: strict
# frozen_string_literal: true

require "rubocops/shared/helper_functions"

module RuboCop
  module Cop
    module Cask
      # This cop checks for the correct order of methods within the 'uninstall' and 'zap' stanzas.
      class UninstallMethodsOrder < Base
        extend AutoCorrector
        include CaskHelp
        include HelperFunctions

        MSG = T.let("`%<method>s` method out of order".freeze, String)

        sig { params(node: AST::SendNode).void }
        def on_send(node)
          return unless [:zap, :uninstall].include?(node.method_name)

          hash_node = node.arguments.first
          return if hash_node.nil? || (!hash_node.is_a?(AST::Node) && !hash_node.hash_type?)

          method_nodes = hash_node.pairs.map(&:key)
          expected_order = method_nodes.sort_by { |method| method_order_index(method) }

          method_nodes.each_with_index do |method, index|
            next if method == expected_order[index]

            add_offense(method, message: format(MSG, method: method.children.first)) do |corrector|
              indentation = " " * (start_column(method) - line_start_column(method))
              ordered_sources = expected_order.map do |expected_method|
                hash_node.pairs.find { |pair| pair.key == expected_method }.source
              end
              new_code = ordered_sources.join(",\n#{indentation}")
              corrector.replace(hash_node.source_range, new_code)
            end
          end
        end

        sig { params(method_node: AST::SymbolNode).returns(Integer) }
        def method_order_index(method_node)
          method_name = method_node.children.first
          RuboCop::Cask::Constants::UNINSTALL_METHODS_ORDER.index(method_name) || -1
        end
      end
    end
  end
end
