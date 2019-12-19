# frozen_string_literal: true

require "formula"
require "tab"
require "diagnostic"
require "cli/parser"

module Homebrew
  module_function

  def missing_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `missing` [<options>] [<formula>]

        Check the given <formula> kegs for missing dependencies. If no <formula> are
        provided, check all kegs. Will exit with a non-zero status if any kegs are found
        to be missing dependencies.
      EOS
      comma_array "--hide",
                  description: "Act as if none of the specified <hidden> are installed. <hidden> should be "\
                               "a comma-separated list of formulae."
      switch :verbose
      switch :debug
    end
  end

  def missing
    missing_args.parse

    return unless HOMEBREW_CELLAR.exist?

    ff = if Homebrew.args.named.blank?
      Formula.installed.sort
    else
      Homebrew.args.resolved_formulae.sort
    end

    ff.each do |f|
      missing = f.missing_dependencies(hide: args.hide)
      next if missing.empty?

      Homebrew.failed = true
      print "#{f}: " if ff.size > 1
      puts missing.join(" ")
    end
  end
end
