# typed: true
# frozen_string_literal: true

require "cli/parser"

module Homebrew
  extend T::Sig

  module_function

  sig { returns(CLI::Parser) }
  def untap_args
    Homebrew::CLI::Parser.new do
      description <<~EOS
        Remove a tapped formula repository.
      EOS
      switch "-f", "--force",
             description: "Untap even if formulae or casks from this tap are currently installed."

      named_args :tap, min: 1
    end
  end

  def untap
    args = untap_args.parse

    args.named.to_installed_taps.each do |tap|
      odie "Untapping #{tap} is not allowed" if tap.core_tap? && !Homebrew::EnvConfig.install_from_api?

      if !Homebrew::EnvConfig.install_from_api? || (!tap.core_tap? && tap != "homebrew/cask")
        installed_tap_formulae = Formula.installed.select { |formula| formula.tap == tap }
        installed_tap_casks = Cask::Caskroom.casks.select { |cask| cask.tap == tap }

        if installed_tap_formulae.present? || installed_tap_casks.present?
          installed_names = (installed_tap_formulae + installed_tap_casks.map(&:token)).join("\n")
          if args.force? || Homebrew::EnvConfig.developer?
            opoo <<~EOS
              Untapping #{tap} even though it contains the following installed formulae or casks:
              #{installed_names}
            EOS
          else
            odie <<~EOS
              Refusing to untap #{tap} because it contains the following installed formulae or casks:
              #{installed_names}
            EOS
          end
        end
      end

      tap.uninstall manual: true
    end
  end
end
