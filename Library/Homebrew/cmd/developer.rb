# typed: strict
# frozen_string_literal: true

require "abstract_command"

module Homebrew
  module Cmd
    class Developer < AbstractCommand
      cmd_args do
        description <<~EOS
          Control Homebrew's developer mode. When developer mode is enabled,
          `brew update` will update Homebrew to the latest commit on the `master`
          branch instead of the latest stable version along with some other behaviour changes.

          `brew developer` [`state`]:
          Display the current state of Homebrew's developer mode.

          `brew developer` (`on`|`off`):
          Turn Homebrew's developer mode on or off respectively.
        EOS

        named_args %w[state on off], max: 1
      end

      sig { override.void }
      def run
        env_vars = []
        env_vars << "HOMEBREW_DEVELOPER" if Homebrew::EnvConfig.developer?
        env_vars << "HOMEBREW_UPDATE_TO_TAG" if Homebrew::EnvConfig.update_to_tag?
        env_vars.map! do |var|
          "#{Tty.bold}#{var}#{Tty.reset}"
        end

        case args.named.first
        when nil, "state"
          if env_vars.any?
            verb = (env_vars.count == 1) ? "is" : "are"
            puts "Developer mode is enabled because #{env_vars.to_sentence} #{verb} set."
          elsif Homebrew::Settings.read("devcmdrun") == "true"
            puts "Developer mode is enabled."
          else
            puts "Developer mode is disabled."
          end
        when "on"
          Homebrew::Settings.write "devcmdrun", true
        when "off"
          Homebrew::Settings.delete "devcmdrun"
          puts "To fully disable developer mode, you must unset #{env_vars.to_sentence}." if env_vars.any?
        else
          raise UsageError, "unknown subcommand: #{args.named.first}"
        end
      end
    end
  end
end
