# frozen_string_literal: true

module Unparser
  class Emitter
    # Emitter for rescue nodes
    class Rescue < self
      handle :rescue

    private

      def dispatch
        emit_rescue_postcontrol(node)
      end
    end # Rescue
  end # Emitter
end # Unparser
