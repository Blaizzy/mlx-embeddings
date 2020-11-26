# typed: true
# frozen_string_literal: true

module Language
  module Haskell
    module Cabal
      def self.included(_)
        odisabled "include Language::Haskell::Cabal"
      end
    end
  end
end
