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
    end
  end
end
