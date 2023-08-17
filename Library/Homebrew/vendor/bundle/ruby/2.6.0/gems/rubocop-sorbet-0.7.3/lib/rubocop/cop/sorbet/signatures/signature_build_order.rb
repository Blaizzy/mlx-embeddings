# frozen_string_literal: true

require "rubocop"
require_relative "signature_cop"

begin
  require "unparser"
rescue LoadError
  nil
end

module RuboCop
  module Cop
    module Sorbet
      class SignatureBuildOrder < SignatureCop
        ORDER =
          [
            :abstract,
            :override,
            :overridable,
            :type_parameters,
            :params,
            :returns,
            :void,
            :soft,
            :checked,
            :on_failure,
          ].each_with_index.to_h.freeze

        # @!method root_call(node)
        def_node_search(:root_call, <<~PATTERN)
          (send nil? {#{ORDER.keys.map(&:inspect).join(" ")}} ...)
        PATTERN

        def on_signature(node)
          calls = call_chain(node.children[2]).map(&:method_name)
          return if calls.empty?

          # While the developer is typing, we may have an incomplete call statement, which means `ORDER[call]` will
          # return `nil`. In that case, invoking `sort_by` will raise
          return if calls.any? { |call| ORDER[call].nil? }

          expected_order = calls.sort_by { |call| ORDER[call] }
          return if expected_order == calls

          message = "Sig builders must be invoked in the following order: #{expected_order.join(", ")}."

          unless can_autocorrect?
            message += " For autocorrection, add the `unparser` gem to your project."
          end

          add_offense(
            node.children[2],
            message: message,
          )
          node
        end

        def autocorrect(node)
          return unless can_autocorrect?

          lambda do |corrector|
            tree = call_chain(node_reparsed_with_modern_features(node))
              .sort_by { |call| ORDER[call.method_name] }
              .reduce(nil) do |receiver, caller|
                caller.updated(nil, [receiver] + caller.children.drop(1))
              end

            corrector.replace(
              node,
              Unparser.unparse(tree),
            )
          end
        end

        # Create a subclass of AST Builder that has modern features turned on
        class ModernBuilder < RuboCop::AST::Builder
          modernize
        end
        private_constant :ModernBuilder

        private

        # This method exists to reparse the current node with modern features enabled.
        # Modern features include "index send" emitting, which is necessary to unparse
        # "index sends" (i.e. `[]` calls) back to index accessors (i.e. as `foo[bar]``).
        # Otherwise, we would get the unparsed node as `foo.[](bar)`.
        def node_reparsed_with_modern_features(node)
          # Create a new parser with a modern builder class instance
          parser = Parser::CurrentRuby.new(ModernBuilder.new)
          # Create a new source buffer with the node source
          buffer = Parser::Source::Buffer.new(processed_source.path, source: node.source)
          # Re-parse the buffer
          parser.parse(buffer)
        end

        def can_autocorrect?
          defined?(::Unparser)
        end

        def call_chain(sig_child_node)
          return [] if sig_child_node.nil?

          call_node = root_call(sig_child_node).first
          return [] unless call_node

          calls = []
          while call_node != sig_child_node
            calls << call_node
            call_node = call_node.parent
          end

          calls << sig_child_node

          calls
        end
      end
    end
  end
end
