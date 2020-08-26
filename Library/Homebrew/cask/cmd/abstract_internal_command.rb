# frozen_string_literal: true

module Cask
  class Cmd
    # Abstract superclass for all internal `brew cask` commands.
    #
    # @api private
    class AbstractInternalCommand < AbstractCommand
      def self.command_name
        super.sub(/^internal_/i, "_")
      end

      def self.visible?
        false
      end
    end
  end
end
