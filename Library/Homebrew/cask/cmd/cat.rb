# frozen_string_literal: true

module Cask
  class Cmd
    class Cat < AbstractCommand
      def initialize(*)
        super
        raise CaskUnspecifiedError if args.empty?
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

      def self.help
        "dump raw source of the given Cask to the standard output"
      end
    end
  end
end
