# frozen_string_literal: true

module Unparser
  class Emitter
    # Emitter for splats
    class KwSplat < self
      handle :kwsplat

      children :subject

    private

      def dispatch
        write('**')
        visit(subject)
      end
    end

    # Emitter for splats
    class Splat < self
      handle :splat

      children :subject

      def emit_mlhs
        write('*')
        subject_emitter.emit_mlhs if subject
      end

    private

      def dispatch
        write('*')
        subject_emitter.write_to_buffer
      end

      def subject_emitter
        emitter(subject)
      end
      memoize :subject_emitter
    end
  end
end # Unparser
