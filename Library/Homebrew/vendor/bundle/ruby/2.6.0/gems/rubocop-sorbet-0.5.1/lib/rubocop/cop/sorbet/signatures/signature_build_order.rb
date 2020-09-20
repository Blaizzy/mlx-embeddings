# frozen_string_literal: true

require 'rubocop'
require_relative 'signature_cop'

begin
  require 'unparser'
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

        def_node_search(:root_call, <<~PATTERN)
          (send nil? {#{ORDER.keys.map(&:inspect).join(' ')}} ...)
        PATTERN

        def on_signature(node)
          calls = call_chain(node.children[2]).map(&:method_name)
          return unless calls.any?

          expected_order = calls.sort_by { |call| ORDER[call] }
          return if expected_order == calls

          message = "Sig builders must be invoked in the following order: #{expected_order.join(', ')}."

          unless can_autocorrect?
            message += ' For autocorrection, add the `unparser` gem to your project.'
          end

          add_offense(
            node.children[2],
            message: message,
          )
          node
        end

        def autocorrect(node)
          return nil unless can_autocorrect?

          lambda do |corrector|
            tree = call_chain(node_with_index_sends(node))
              .sort_by { |call| ORDER[call.method_name] }
              .reduce(nil) do |receiver, caller|
                caller.updated(nil, [receiver] + caller.children.drop(1))
              end

            corrector.replace(
              node.source_range,
              Unparser.unparse(tree),
            )
          end
        end

        private

        def node_with_index_sends(node)
          # This is really dirty hack to reparse the current node with index send
          # emitting enabled, which is necessary to unparse them back as index accessors.
          emit_index_value = RuboCop::AST::Builder.emit_index
          RuboCop::AST::Builder.emit_index = true
          RuboCop::AST::ProcessedSource.new(node.source, target_ruby_version, processed_source.path).ast
        ensure
          RuboCop::AST::Builder.emit_index = emit_index_value
        end

        def can_autocorrect?
          defined?(::Unparser)
        end

        def call_chain(sig_child_node)
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
