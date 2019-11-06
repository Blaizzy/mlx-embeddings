# frozen_string_literal: true

require "cli/parser"

module Homebrew
  module_function

  def tap_pin_args
    Homebrew::CLI::Parser.new do
      hide_from_man_page!
    end
  end

  def tap_pin
    odisabled "brew tap-pin user/tap",
              "fully-scoped user/tap/formula naming"
  end
end
