# frozen_string_literal: true

require "cli/parser"

module Homebrew
  module_function

  def analytics_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `analytics` [<subcommand>]

        If `on` or `off` is passed, turn Homebrew's analytics on or off respectively.

        If `state` is passed, display the current anonymous user behaviour analytics state.
        Read more at <https://docs.brew.sh/Analytics>.

        If `regenerate-uuid` is passed, regenerate the UUID used in Homebrew's analytics.
      EOS
      switch :verbose
      switch :debug
      max_named 1
    end
  end

  def analytics
    analytics_args.parse

    case args.remaining.first
    when nil, "state"
      if Utils::Analytics.disabled?
        puts "Analytics are disabled."
      else
        puts "Analytics are enabled."
        puts "UUID: #{Utils::Analytics.uuid}" if Utils::Analytics.uuid.present?
      end
    when "on"
      Utils::Analytics.enable!
    when "off"
      Utils::Analytics.disable!
    when "regenerate-uuid"
      Utils::Analytics.regenerate_uuid!
    else
      raise UsageError, "Unknown subcommand."
    end
  end
end
