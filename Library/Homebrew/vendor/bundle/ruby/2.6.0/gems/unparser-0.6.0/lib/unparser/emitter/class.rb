# frozen_string_literal: true

module Unparser
  class Emitter
    # Emitter for class nodes
    class Class < self
      include LocalVariableRoot

      handle :class

      children :name, :superclass, :body

    private

      def dispatch
        write('class ')
        visit(name)
        emit_superclass
        emit_optional_body(body)
        k_end
      end

      def emit_superclass
        return unless superclass

        write(' < ')
        visit(superclass)
      end

    end # Class

    # Emitter for sclass nodes
    class SClass < self
      handle :sclass

      children :object, :body

    private

      def dispatch
        write('class << ')
        visit(object)
        emit_optional_body(body)
        k_end
      end

    end # SClass
  end # Emitter
end # Unparser
