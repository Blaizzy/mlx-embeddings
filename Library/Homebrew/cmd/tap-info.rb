# frozen_string_literal: true

require "cli/parser"

module Homebrew
  module_function

  def tap_info_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `tap-info` [<options>] [<tap>]

        Show detailed information about one or more <tap>s.

        If no <tap> names are provided, display brief statistics for all installed taps.
      EOS
      switch "--installed",
             description: "Show information on each installed tap."
      flag   "--json",
             description: "Print a JSON representation of <tap>. Currently the default and only accepted "\
                          "value for <version> is `v1`. See the docs for examples of using the JSON "\
                          "output: <https://docs.brew.sh/Querying-Brew>"
      switch :debug
    end
  end

  def tap_info
    tap_info_args.parse

    if args.installed?
      taps = Tap
    else
      taps = Homebrew.args.named.sort.map do |name|
        Tap.fetch(name)
      end
    end

    if args.json
      raise UsageError, "Invalid JSON version: #{args.json}" unless ["v1", true].include? args.json

      print_tap_json(taps.sort_by(&:to_s))
    else
      print_tap_info(taps.sort_by(&:to_s))
    end
  end

  def print_tap_info(taps)
    if taps.none?
      tap_count = 0
      formula_count = 0
      command_count = 0
      pinned_count = 0
      private_count = 0
      Tap.each do |tap|
        tap_count += 1
        formula_count += tap.formula_files.size
        command_count += tap.command_files.size
        pinned_count += 1 if tap.pinned?
        private_count += 1 if tap.private?
      end
      info = "#{tap_count} #{"tap".pluralize(tap_count)}"
      info += ", #{pinned_count} pinned"
      info += ", #{private_count} private"
      info += ", #{formula_count} #{"formula".pluralize(formula_count)}"
      info += ", #{command_count} #{"command".pluralize(command_count)}"
      info += ", #{Tap::TAP_DIRECTORY.dup.abv}" if Tap::TAP_DIRECTORY.directory?
      puts info
    else
      taps.each_with_index do |tap, i|
        puts unless i.zero?
        info = "#{tap}: "
        if tap.installed?
          info += tap.pinned? ? "pinned" : "unpinned"
          info += ", private" if tap.private?
          info += if (contents = tap.contents).empty?
            ", no commands/casks/formulae"
          else
            ", #{contents.join(", ")}"
          end
          info += "\n#{tap.path} (#{tap.path.abv})"
          info += "\nFrom: #{tap.remote.nil? ? "N/A" : tap.remote}"
        else
          info += "Not installed"
        end
        puts info
      end
    end
  end

  def print_tap_json(taps)
    puts JSON.generate(taps.map(&:to_hash))
  end
end
