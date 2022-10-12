# typed: true
# frozen_string_literal: true

module Homebrew
  module CLI
    class Args
      undef only_formula_or_cask

      def only_formula_or_cask
        # Make formula the default on linux for non-developers
        return :formula unless Homebrew::EnvConfig.developer?
        return :formula if formula? && !cask?
        return :cask if cask? && !formula?
      end
    end
  end
end
