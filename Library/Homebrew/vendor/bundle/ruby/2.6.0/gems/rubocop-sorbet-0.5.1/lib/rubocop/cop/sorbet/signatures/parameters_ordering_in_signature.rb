# frozen_string_literal: true

require 'rubocop'
require_relative 'signature_cop'

module RuboCop
  module Cop
    module Sorbet
      # This cop checks for inconsistent ordering of parameters between the
      # signature and the method definition. The sorbet-runtime gem raises
      # when such inconsistency occurs.
      #
      # @example
      #
      #   # bad
      #   sig { params(a: Integer, b: String).void }
      #   def foo(b:, a:); end
      #
      #   # good
      #   sig { params(a: Integer, b: String).void }
      #   def foo(a:, b:); end
      class ParametersOrderingInSignature < SignatureCop
        def_node_search(:signature_params, <<-PATTERN)
          (send _ :params ...)
        PATTERN

        def on_signature(node)
          sig_params = signature_params(node).first

          sig_params_order = extract_parameters(sig_params)
          return if sig_params_order.nil?
          method_node = node.parent.children[node.sibling_index + 1]
          return if method_node.nil? || method_node.type != :def
          method_parameters = method_node.arguments

          check_for_inconsistent_param_ordering(sig_params_order, method_parameters)
        end

        private

        def extract_parameters(sig_params)
          return [] if sig_params.nil?

          arguments = sig_params.arguments.first
          return arguments.keys.map(&:value) if RuboCop::AST::HashNode === arguments

          add_offense(
            sig_params,
            message: "Invalid signature."
          )
        end

        def check_for_inconsistent_param_ordering(sig_params_order, parameters)
          parameters.each_with_index do |param, index|
            param_name = param.children[0]
            sig_param_name = sig_params_order[index]

            next if param_name == sig_param_name

            add_offense(
              param,
              message: "Inconsistent ordering of arguments at index #{index}. " \
              "Expected `#{sig_param_name}` from sig above."
            )
          end
        end
      end
    end
  end
end
