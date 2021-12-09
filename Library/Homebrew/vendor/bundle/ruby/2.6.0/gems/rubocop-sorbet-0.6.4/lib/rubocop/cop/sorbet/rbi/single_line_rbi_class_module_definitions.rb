# frozen_string_literal: true

module RuboCop
  module Cop
    module Sorbet
      # This cop ensures empty class/module definitions in RBI files are
      # done on a single line rather than being split across multiple lines.
      #
      # @example
      #
      #   # bad
      #   module SomeModule
      #   end
      #
      #   # good
      #   module SomeModule; end
      class SingleLineRbiClassModuleDefinitions < RuboCop::Cop::Cop
        MSG = "Empty class/module definitions in RBI files should be on a single line."

        def on_module(node)
          process_node(node)
        end

        def on_class(node)
          process_node(node)
        end

        def autocorrect(node)
          -> (corrector) { corrector.replace(node, convert_newlines(node.source)) }
        end

        protected

        def convert_newlines(source)
          source.sub(/[\r\n]+\s*[\r\n]*/, "; ")
        end

        def process_node(node)
          return if node.body
          return if node.single_line?
          add_offense(node)
        end
      end
    end
  end
end
