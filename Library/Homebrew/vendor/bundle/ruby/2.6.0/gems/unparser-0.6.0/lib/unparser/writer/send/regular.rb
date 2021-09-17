# frozen_string_literal: true

module Unparser
  module Writer
    class Send
      # Writer for "regular" receiver.selector(arguments...) case
      class Regular < self
        def dispatch
          emit_receiver
          emit_selector
          emit_arguments
        end

        def emit_send_mlhs
          dispatch
        end

        def emit_arguments_without_heredoc_body
          emit_normal_arguments if arguments.any?
        end

        def emit_receiver
          return unless receiver

          emit_send_regular(receiver)

          emit_operator
        end

      end # Regular
    end # Send
  end # Writer
end # Unparser
