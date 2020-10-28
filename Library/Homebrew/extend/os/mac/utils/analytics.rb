# typed: strict
# frozen_string_literal: true

module Utils
  module Analytics
    class << self
      extend T::Sig
      sig { returns(String) }
      def custom_prefix_label
        "non-/usr/local"
      end
    end
  end
end
