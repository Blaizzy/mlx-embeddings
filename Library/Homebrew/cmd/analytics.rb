# typed: true
# frozen_string_literal: true

require "cli/parser"

module Homebrew
  module_function

  sig { returns(CLI::Parser) }
  def analytics_args
    Homebrew::CLI::Parser.new do
      description <<~EOS
        Control Homebrew's anonymous aggregate user behaviour analytics.
        Read more at <https://docs.brew.sh/Analytics>.

        `brew analytics` [`state`]:
        Display the current state of Homebrew's analytics.

        `brew analytics` (`on`|`off`):
        Turn Homebrew's analytics on or off respectively.
      EOS

      named_args %w[state on off regenerate-uuid], max: 1
    end
  end

  def analytics
    args = analytics_args.parse

    case args.named.first
    when nil, "state"
      if Utils::Analytics.disabled?
        puts "Analytics are disabled."
      elsif Homebrew::EnvConfig.no_google_analytics?
        puts "InfluxDB analytics are enabled."
        puts "Google Analytics are disabled."
      else
        puts "Analytics are enabled."
      end
    when "on"
      Utils::Analytics.enable!
    when "off"
      Utils::Analytics.disable!
    when "regenerate-uuid"
      Utils::Analytics.delete_uuid!
      opoo "Homebrew no longer uses an analytics UUID so this has been deleted!"
      puts "brew analytics regenerate-uuid is no longer necessary."
    else
      raise UsageError, "unknown subcommand: #{args.named.first}"
    end
  end
end
