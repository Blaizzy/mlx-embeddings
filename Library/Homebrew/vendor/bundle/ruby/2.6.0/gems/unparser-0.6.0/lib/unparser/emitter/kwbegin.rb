# frozen_string_literal: true

module Unparser
  class Emitter
    # Emitter for explicit begins
    class KWBegin < self
      handle :kwbegin

    private

      def dispatch
        write('begin')

        if children.one?
          emit_body_ensure_rescue(children.first)
        else
          indented do
            emit_multiple_body
          end
        end

        k_end
      end

      def emit_multiple_body
        emit_join(children, method(:emit_body_member), method(:nl))
      end

    end # KWBegin
  end # Emitter
end # Unparser
