require "cli_parser"

module Homebrew
  module_function

  def tap_unpin_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `tap-unpin` <tap>

        Unpin <tap> so its formulae are no longer prioritised. See also `tap-pin`.
      EOS
      switch :debug
      hide_from_man_page!
    end
  end

  def tap_unpin
    odeprecated "brew tap-pin user/tap",
      "fully-scoped user/tap/formula naming"

    tap_unpin_args.parse

    ARGV.named.each do |name|
      tap = Tap.fetch(name)
      raise "unpinning #{tap} is not allowed" if tap.core_tap?

      tap.unpin
      ohai "Unpinned #{tap}"
    end
  end
end
