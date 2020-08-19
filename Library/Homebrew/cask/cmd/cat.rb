# frozen_string_literal: true

module Cask
  class Cmd
    # Implementation of the `brew cask cat` command.
    #
    # @api private
    class Cat < AbstractCommand
      def self.min_named
        :cask
      end

      def self.description
        "Dump raw source of a <cask> to the standard output."
      end

      def run
        casks.each do |cask|
          if Homebrew::EnvConfig.bat?
            ENV["BAT_CONFIG_PATH"] = Homebrew::EnvConfig.bat_config_path
            safe_system "#{HOMEBREW_PREFIX}/bin/bat", cask.sourcefile_path
          else
            puts File.open(cask.sourcefile_path, &:read)
          end
        end
      end
    end
  end
end
