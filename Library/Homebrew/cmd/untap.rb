# typed: true
# frozen_string_literal: true

require "cli/parser"

module Homebrew
  sig { returns(CLI::Parser) }
  def self.untap_args
    Homebrew::CLI::Parser.new do
      description <<~EOS
        Remove a tapped formula repository.
      EOS
      switch "-f", "--force",
             description: "Untap even if formulae or casks from this tap are currently installed."

      named_args :tap, min: 1
    end
  end

  sig { void }
  def self.untap
    args = untap_args.parse

    args.named.to_installed_taps.each do |tap|
      odie "Untapping #{tap} is not allowed" if tap.core_tap? && Homebrew::EnvConfig.no_install_from_api?

      if Homebrew::EnvConfig.no_install_from_api? || (!tap.core_tap? && !tap.core_cask_tap?)
        installed_tap_formulae = installed_formulae_for(tap: tap)
        installed_tap_casks = installed_casks_for(tap: tap)

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

  sig { params(tap: Tap).returns(T::Array[Formula]) }
  def self.installed_formulae_for(tap:)
    tap.formula_names.filter_map do |formula_name|
      next unless installed_formulae_names.include?(T.must(formula_name.split("/").last))

      formula = begin
        Formulary.factory(formula_name)
      rescue
        # Don't blow up because of a single unavailable formula.
        next
      end

      # Can't use Formula#any_version_installed? because it doesn't consider
      # taps correctly.
      formula if formula.installed_kegs.any? { |keg| keg.tab.tap == tap }
    end
  end

  sig { returns(T::Set[String]) }
  def self.installed_formulae_names
    @installed_formulae_names ||= Formula.installed_formula_names.to_set
  end

  sig { params(tap: Tap).returns(T::Array[Cask::Cask]) }
  def self.installed_casks_for(tap:)
    tap.cask_tokens.filter_map do |cask_token|
      next unless installed_cask_tokens.include?(T.must(cask_token.split("/").last))

      cask = begin
        Cask::CaskLoader.load(cask_token)
      rescue
        # Don't blow up because of a single unavailable cask.
        next
      end

      cask if cask.installed?
    end
  end

  sig { returns(T::Set[String]) }
  def self.installed_cask_tokens
    @installed_cask_tokens ||= Cask::Caskroom.tokens.to_set
  end
end
