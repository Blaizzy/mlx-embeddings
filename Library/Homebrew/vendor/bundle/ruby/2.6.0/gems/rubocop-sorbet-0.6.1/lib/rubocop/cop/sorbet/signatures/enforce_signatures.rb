# frozen_string_literal: true

require 'rubocop'
require 'stringio'
require_relative 'signature_cop'

module RuboCop
  module Cop
    module Sorbet
      # This cop checks that every method definition and attribute accessor has a Sorbet signature.
      #
      # It also suggest an autocorrect with placeholders so the following code:
      #
      # ```
      # def foo(a, b, c); end
      # ```
      #
      # Will be corrected as:
      #
      # ```
      # sig { params(a: T.untyped, b: T.untyped, c: T.untyped).returns(T.untyped)
      # def foo(a, b, c); end
      # ```
      #
      # You can configure the placeholders used by changing the following options:
      #
      # * `ParameterTypePlaceholder`: placeholders used for parameter types (default: 'T.untyped')
      # * `ReturnTypePlaceholder`: placeholders used for return types (default: 'T.untyped')
      class EnforceSignatures < SignatureCop
        def initialize(config = nil, options = nil)
          super(config, options)
          @last_sig_for_scope = {}
        end

        def_node_matcher(:accessor?, <<-PATTERN)
          (send nil? {:attr_reader :attr_writer :attr_accessor} ...)
        PATTERN

        def on_def(node)
          check_node(node)
        end

        def on_defs(node)
          check_node(node)
        end

        def on_send(node)
          check_node(node) if accessor?(node)
        end

        def on_block(node)
          @last_sig_for_scope[scope(node)] = node if signature?(node)
        end

        def autocorrect(node)
          lambda do |corrector|
            suggest = SigSuggestion.new(node.loc.column, param_type_placeholder, return_type_placeholder)

            if node.is_a?(RuboCop::AST::DefNode) # def something
              node.arguments.each do |arg|
                suggest.params << arg.children.first
              end
            elsif accessor?(node) # attr reader, writer, accessor
              method = node.children[1]
              symbol = node.children[2]
              suggest.params << symbol.value if symbol && (method == :attr_writer || method == :attr_accessor)
              suggest.returns = 'void' if method == :attr_writer
            end

            corrector.insert_before(node.loc.expression, suggest.to_autocorrect)
          end
        end

        def scope(node)
          return nil unless node.parent
          return node.parent if [:begin, :block, :class, :module].include?(node.parent.type)
          scope(node.parent)
        end

        private

        def check_node(node)
          scope = self.scope(node)
          unless @last_sig_for_scope[scope]
            add_offense(
              node,
              message: "Each method is required to have a signature."
            )
          end
          @last_sig_for_scope[scope] = nil
        end

        def param_type_placeholder
          cop_config['ParameterTypePlaceholder'] || 'T.untyped'
        end

        def return_type_placeholder
          cop_config['ReturnTypePlaceholder'] || 'T.untyped'
        end

        class SigSuggestion
          attr_accessor :params, :returns

          def initialize(indent, param_placeholder, return_placeholder)
            @params = []
            @returns = nil
            @indent = indent
            @param_placeholder = param_placeholder
            @return_placeholder = return_placeholder
          end

          def to_autocorrect
            out = StringIO.new
            out << 'sig { '
            out << generate_params
            out << generate_return
            out << " }\n"
            out << ' ' * @indent # preserve indent for the next line
            out.string
          end

          private

          def generate_params
            return if @params.empty?
            out = StringIO.new
            out << 'params('
            out << @params.map do |param|
              "#{param}: #{@param_placeholder}"
            end.join(", ")
            out << ').'
            out.string
          end

          def generate_return
            return "returns(#{@return_placeholder})" if @returns.nil?
            return @returns if @returns == 'void'
            "returns(#{@returns})"
          end
        end
      end
    end
  end
end
