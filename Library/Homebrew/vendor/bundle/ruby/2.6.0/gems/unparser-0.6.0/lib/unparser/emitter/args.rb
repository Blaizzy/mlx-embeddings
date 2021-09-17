# frozen_string_literal: true

module Unparser
  class Emitter
    # Arguments emitter
    class Args < self
      def emit_block_arguments
        delimited(normal_arguments)

        write(',') if normal_arguments.one? && n_arg?(normal_arguments.first)

        emit_shadowargs
      end

      def emit_def_arguments
        delimited(normal_arguments)
      end

      def emit_lambda_arguments
        delimited(normal_arguments)
        emit_shadowargs
      end

    private

      def emit_shadowargs
        return if shadowargs.empty?

        write('; ')
        delimited(shadowargs)
      end

      def normal_arguments
        children.reject(&method(:n_shadowarg?))
      end
      memoize :normal_arguments

      def shadowargs
        children.select(&method(:n_shadowarg?))
      end
      memoize :shadowargs

    end # Arguments
  end # Emitter
end # Unparser
