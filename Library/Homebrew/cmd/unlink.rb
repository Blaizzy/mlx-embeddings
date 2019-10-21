# frozen_string_literal: true

require "ostruct"
require "cli/parser"

module Homebrew
  module_function

  def unlink_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `unlink` [<options>] <formula>

        Remove symlinks for <formula> from Homebrew's prefix. This can be useful
        for temporarily disabling a formula:
        `brew unlink` <formula> `&&` <commands> `&& brew link` <formula>
      EOS
      switch "-n", "--dry-run",
             description: "List files which would be unlinked without actually unlinking or "\
                          "deleting any files."
      switch :verbose
      switch :debug
    end
  end

  def unlink
    unlink_args.parse

    raise KegUnspecifiedError if args.remaining.empty?

    mode = OpenStruct.new
    mode.dry_run = true if args.dry_run?

    Homebrew.args.kegs.each do |keg|
      if mode.dry_run
        puts "Would remove:"
        keg.unlink(mode)
        next
      end

      keg.lock do
        print "Unlinking #{keg}... "
        puts if args.verbose?
        puts "#{keg.unlink(mode)} symlinks removed"
      end
    end
  end
end
