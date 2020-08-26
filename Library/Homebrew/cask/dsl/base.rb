# frozen_string_literal: true

require "cask/utils"

module Cask
  class DSL
    # Superclass for all stanzas which take a block.
    #
    # @api private
    class Base
      extend Forwardable

      def initialize(cask, command = SystemCommand)
        @cask = cask
        @command = command
      end

      def_delegators :@cask, :token, :version, :caskroom_path, :staged_path, :appdir, :language

      def system_command(executable, **options)
        @command.run!(executable, **options)
      end

      def respond_to_missing?(*)
        super
      end

      def method_missing(method, *)
        if method
          underscored_class = self.class.name.gsub(/([[:lower:]])([[:upper:]][[:lower:]])/, '\1_\2').downcase
          section = underscored_class.split("::").last
          Utils.method_missing_message(method, @cask.to_s, section)
          nil
        else
          super
        end
      end
    end
  end
end
