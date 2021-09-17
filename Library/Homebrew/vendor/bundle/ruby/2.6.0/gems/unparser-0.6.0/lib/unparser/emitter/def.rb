# frozen_string_literal: true

module Unparser
  class Emitter
    # Emitter for def node
    class Def < self
      include LocalVariableRoot

    private

      abstract_method :emit_name
      private :emit_name

      abstract_method :body
      private :body

      def dispatch
        write('def ')
        emit_name
        emit_arguments
        emit_optional_body_ensure_rescue(body)
        k_end
      end

      def emit_arguments
        return if arguments.children.empty?

        parentheses do
          writer_with(Args, arguments).emit_def_arguments
        end
      end

      # Instance def emitter
      class Instance < self
        handle :def

        children :name, :arguments, :body

      private

        def emit_name
          write(name.to_s)
        end

      end # Instance

      # Emitter for defines on singleton
      class Singleton < self

        handle :defs

        children :subject, :name, :arguments, :body

      private

        def emit_name
          conditional_parentheses(!subject_without_parens?) do
            visit(subject)
          end
          write('.', name.to_s)
        end

        def subject_without_parens?
          case subject.type
          when :self
            true
          when :const
            !subject.children.first
          when :send
            receiver, _selector, *arguments = *subject
            !receiver && arguments.empty?
          end
        end

      end # Singleton
    end # Def
  end # Emitter
end # Unparser
