# frozen_string_literal: true

module Cask
  class Cmd
    class Home < AbstractCommand
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

      def self.help
        "opens the homepage of the given Cask"
      end
    end
  end
end
