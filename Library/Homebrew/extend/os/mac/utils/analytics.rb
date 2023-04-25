# typed: strict
# frozen_string_literal: true

module Utils
  module Analytics
    class << self
      sig { returns(String) }
      def custom_prefix_label_google
        return generic_custom_prefix_label_google if Hardware::CPU.arm?

        "non-/usr/local"
      end

      sig { returns(String) }
      def arch_label_google
        return "Rosetta" if Hardware::CPU.in_rosetta2?

        generic_arch_label_google
      end
    end
  end
end
