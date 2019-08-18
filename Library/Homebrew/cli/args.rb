# frozen_string_literal: true

require "ostruct"

module Homebrew
  module CLI
    class Args < OpenStruct
      # undefine tap to allow --tap argument
      undef tap

      def initialize(argv:)
        super
        @argv = argv
      end

      def to_cli_option(name)
        if name.length == 2
          "-#{name.tr("?", "")}"
        else
          "--#{name.tr("_", "-").tr("?", "")}"
        end
      end

      def options_only
        to_h.keys
            .map(&:to_s)
            .reject { |name| %w[argv remaining].include?(name) }
            .map(&method(:to_cli_option))
            .select { |arg| arg.start_with?("-") }
      end
    end
  end
end
