# frozen_string_literal: true

module Unparser
  class Emitter
    # Emitter for block and kwrestarg arguments
    class Morearg < self
      MAP = {
        blockarg:  '&',
        kwrestarg: '**'
      }.freeze

      handle(*MAP.keys)

      children :name

    private

      def dispatch
        write(MAP.fetch(node_type), name.to_s)
      end

    end # Blockarg

    # Optional argument emitter
    class Optarg < self
      handle :optarg

      children :name, :value

    private

      def dispatch
        write(name.to_s, ' = ')
        visit(value)
      end
    end

    # Optional keyword argument emitter
    class KeywordOptional < self
      handle :kwoptarg

      children :name, :value

    private

      def dispatch
        write(name.to_s, ': ')
        visit(value)
      end

    end # KeywordOptional

    # Keyword argument emitter
    class Kwarg < self
      handle :kwarg

      children :name

    private

      def dispatch
        write(name.to_s, ':')
      end

    end # Restarg

    # Rest argument emitter
    class Restarg < self
      handle :restarg

      children :name

    private

      def dispatch
        write('*', name.to_s)
      end

    end # Restarg

    # Argument emitter
    class Argument < self
      handle :arg, :shadowarg

      children :name

    private

      def dispatch
        write(name.to_s)
      end

    end # Argument

    # Progarg emitter
    class Procarg < self
      handle :procarg0

      PARENS = %i[restarg mlhs].freeze

    private

      def dispatch
        if needs_parens?
          parentheses do
            delimited(children)
          end
        else
          delimited(children)
        end
      end

      def needs_parens?
        children.length > 1 || children.any? do |node|
          PARENS.include?(node.type)
        end
      end
    end

    # Block pass node emitter
    class BlockPass < self
      handle :block_pass

      children :name

    private

      def dispatch
        write('&')
        visit(name)
      end

    end # BlockPass

  end # Emitter
end # Unparser
