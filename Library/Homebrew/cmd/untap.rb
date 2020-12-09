# typed: true
# frozen_string_literal: true

require "cli/parser"
require "formula"

module Homebrew
  extend T::Sig

  module_function

  sig { returns(CLI::Parser) }
  def untap_args
    Homebrew::CLI::Parser.new do
      usage_banner <<~EOS
        `untap` <tap>

        Remove a tapped formula repository.
      EOS
      switch "-f", "--force",
             description: "Untap even if formulae or casks from this tap are currently installed."

      min_named 1
    end
  end

  def untap
    args = untap_args.parse

    args.named.each do |tapname|
      tap = Tap.fetch(tapname)
      odie "Untapping #{tap} is not allowed" if tap.core_tap?

      installed_tap_formulae = Formula.installed.select { |formula| formula.tap == tap }
      installed_tap_casks = Cask::Caskroom.casks.select { |cask| cask.tap == tap }

      if installed_tap_formulae.length.positive? || installed_tap_casks.length.positive?
        if args.force?
          opoo <<~EOS
            Untapping #{tap} even though it contains the following installed formulae or casks:
            #{(installed_tap_formulae + installed_tap_casks.map(&:token)).join("\n")}
          EOS
        else
          odie <<~EOS
            Refusing to untap #{tap} because it contains the following installed formulae or casks:
            #{(installed_tap_formulae + installed_tap_casks.map(&:token)).join("\n")}
          EOS
        end
      end

      tap.uninstall
    end
  end
end
