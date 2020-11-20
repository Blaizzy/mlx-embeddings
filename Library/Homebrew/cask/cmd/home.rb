# typed: true
# frozen_string_literal: true

module Cask
  class Cmd
    # Implementation of the `brew cask home` command.
    #
    # @api private
    class Home < AbstractCommand
      extend T::Sig

      sig { returns(String) }
      def self.description
        "Opens the homepage of the given <cask>. If no cask is given, opens the Homebrew homepage."
      end

      sig { void }
      def run
        if casks.none?
          odebug "Opening project homepage"
          self.class.open_url "https://brew.sh/"
        else
          casks.each do |cask|
            odebug "Opening homepage for Cask #{cask}"
            self.class.open_url cask.homepage
          end
        end
      end

      def self.open_url(url)
        SystemCommand.run!(OS::PATH_OPEN, args: ["--", url])
      end
    end
  end
end
